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
"""Online training script that collects data from environment and trains the policy.

This script combines data collection (similar to lerobot_eval.py) with training (similar to lerobot_train.py).
It alternates between:
1. Collecting episodes from the environment using the current policy
2. Adding collected data to the dataset (placeholder for user implementation)
3. Training the policy on the accumulated dataset

Usage examples:

```
lerobot-online-train \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --dataset.repo_id=my_online_dataset \
    --online.collect_episodes_per_iteration=10 \
    --online.train_steps_per_iteration=1000 \
    --online.n_iterations=100 \
    --policy.use_amp=false \
    --policy.device=cuda
```
"""

import logging
import time
from contextlib import nullcontext
from collections.abc import Callable
from pathlib import Path
from pprint import pformat
from typing import Any
from tqdm import trange
from copy import deepcopy

import einops
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.configs import parser
from lerobot.configs.train import OnlineTrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import _compile_episode_data
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)


class FillMissingActionContextProcessor:
    """为缺失的动作上下文字段填充默认值。
    
    用于处理离线数据集不包含 prev_actions、pred_action、actions_seq_valid 字段的情况。
    当这些字段缺失时，用零填充并将 valid 标志设为 False。
    """
    
    def __init__(self, action_dim: int, prev_steps: int = 10, pred_steps: int = 10):
        """初始化 Processor。
        
        Args:
            action_dim: 动作维度
            prev_steps: prev_actions 的步数（默认10）
            pred_steps: pred_action 的步数（默认10）
        """
        self.action_dim = action_dim
        self.prev_steps = prev_steps
        self.pred_steps = pred_steps
    
    def __call__(self, batch: dict) -> dict:
        """处理 batch，为缺失字段填充默认值。
        
        Args:
            batch: 数据 batch
            
        Returns:
            填充后的 batch
        """
        # 获取 batch_size 和 device
        sample_tensor = None
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                sample_tensor = v
                break
        
        if sample_tensor is None:
            return batch
        
        batch_size = sample_tensor.shape[0]
        device = sample_tensor.device
        
        if 'prev_actions' not in batch:
            batch['prev_actions'] = torch.zeros(
                batch_size, self.prev_steps, self.action_dim,
                device=device, dtype=torch.float32
            )
        
        if 'pred_action' not in batch:
            batch['pred_action'] = torch.zeros(
                batch_size, self.pred_steps, self.action_dim,
                device=device, dtype=torch.float32
            )
        
        if 'actions_seq_valid' not in batch:
            batch['actions_seq_valid'] = torch.zeros(
                batch_size, 1, device=device, dtype=torch.bool
            )
        
        return batch


class ValidActionContextSampler(torch.utils.data.Sampler):
    """只采样 actions_seq_valid=True 的帧的 Sampler。"""

    def __init__(self, dataset: LeRobotDataset, shuffle: bool = True):
        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self._valid_indices = None
        self._cached_num_frames = 0

    def _find_valid_indices(self) -> list[int]:
        """高效地查找所有 actions_seq_valid=True 的帧索引。"""

        # 检查数据集是否有 actions_seq_valid 字段
        if 'actions_seq_valid' not in self.dataset.features:
            logging.warning(
                "Dataset does not have 'actions_seq_valid' feature. "
                "Falling back to all indices."
            )
            return list(range(len(self.dataset)))

        logging.info(f"Finding valid action context frames (total: {len(self.dataset)} frames)...")

        # 直接从 hf_dataset 批量读取 actions_seq_valid 列（非常快）
        hf_dataset = self.dataset.hf_dataset
        if 'actions_seq_valid' in hf_dataset.column_names:
            # 批量获取整列数据
            actions_seq_valid = np.array(hf_dataset['actions_seq_valid'])
            if len(actions_seq_valid) == 0:
                return []
            # 转换为 numpy 数组进行快速过滤
            if isinstance(actions_seq_valid, list):
                actions_seq_valid = np.array(actions_seq_valid)

            # 处理形状：可能是 (N,) 或 (N, 1)
            if actions_seq_valid.ndim == 2:
                actions_seq_valid = actions_seq_valid.squeeze(-1)

            # 找到所有 True 的索引
            valid_indices = np.where(actions_seq_valid)[0].tolist()
        else:
            # 如果 hf_dataset 中没有这个列，回退到全部索引
            logging.warning("'actions_seq_valid' not in hf_dataset columns, using all indices")
            valid_indices = list(range(len(self.dataset)))

        logging.info(
            f"Found {len(valid_indices)} valid frames out of {len(self.dataset)} "
            f"({100 * len(valid_indices) / max(1, len(self.dataset)):.1f}%)"
        )

        return valid_indices

    @property
    def valid_indices(self) -> list[int]:
        """获取有效帧索引（带缓存）。"""
        current_num_frames = len(self.dataset)
        if self._valid_indices is None or self._cached_num_frames != current_num_frames:
            self._valid_indices = self._find_valid_indices()
            self._cached_num_frames = current_num_frames
        return self._valid_indices

    def __iter__(self):
        indices = self.valid_indices
        if len(indices) == 0:
            logging.warning("No valid frames found! Returning empty iterator.")
            return iter([])

        if self.shuffle:
            perm = torch.randperm(len(indices)).tolist()
            return iter([indices[i] for i in perm])
        return iter(indices)

    def __len__(self) -> int:
        return len(self.valid_indices)

# class ValidActionContextSampler(torch.utils.data.Sampler):
#      """只采样 actions_seq_valid=True 的帧的 Sampler。
#
#      用于在线数据集训练时，仅使用动作上下文有效的帧，
#      即 prev_actions 和 pred_action 来自同一次预测的帧。
#      """
#
#      def __init__(self, dataset: LeRobotDataset, shuffle: bool = True):
#          """初始化 Sampler。
#
#          Args:
#              dataset: LeRobotDataset 实例
#              shuffle: 是否打乱采样顺序（默认 True）
#          """
#          self.dataset = dataset
#          self.shuffle = shuffle
#          self._valid_indices = None
#          self._cached_num_frames = 0
#
#      def _find_valid_indices(self) -> list[int]:
#          """查找所有 actions_seq_valid=True 的帧索引。
#
#          Returns:
#              有效帧的索引列表
#          """
#          valid_indices = []
#
#          # 检查数据集是否有 actions_seq_valid 字段
#          if 'actions_seq_valid' not in self.dataset.features:
#              logging.warning(
#                  "Dataset does not have 'actions_seq_valid' feature. "
#                  "Falling back to all indices."
#              )
#              return list(range(len(self.dataset)))
#          logging.info(f"Scanning dataset for valid action context frames (total: {len(self.dataset)} frames)...")
#
#          for idx in range(len(self.dataset)):
#              try:
#                  item = self.dataset[idx]
#                  if 'actions_seq_valid' in item:
#                      valid_value = item['actions_seq_valid']
#                      # 处理不同格式：可能是 [1] 形状的张量或标量
#                      if isinstance(valid_value, torch.Tensor):
#                          if valid_value.numel() == 1:
#                              is_valid = valid_value.item()
#                          else:
#                              is_valid = valid_value.any().item()
#                      else:
#                          is_valid = bool(valid_value)
#
#                      if is_valid:
#                          valid_indices.append(idx)
#              except Exception as e:
#                  logging.warning(f"Error reading frame {idx}: {e}")
#                  continue
#          logging.info(
#              f"Found {len(valid_indices)} valid frames out of {len(self.dataset)} "
#              f"({100 * len(valid_indices) / max(1, len(self.dataset)):.1f}%)"
#          )
#
#          return valid_indices
#
#      @property
#      def valid_indices(self) -> list[int]:
#          """获取有效帧索引（带缓存）。"""
#          # 如果数据集大小变化了，需要重新扫描
#          current_num_frames = len(self.dataset)
#          if self._valid_indices is None or self._cached_num_frames != current_num_frames:
#              self._valid_indices = self._find_valid_indices()
#              self._cached_num_frames = current_num_frames
#          return self._valid_indices
#
#
#      def __iter__(self):
#          indices = self.valid_indices
#          if len(indices) == 0:
#              logging.warning("No valid frames found! Returning empty iterator.")
#              return iter([])
#
#          if self.shuffle:
#              # 使用 torch.randperm 生成随机排列
#              perm = torch.randperm(len(indices)).tolist()
#              return iter([indices[i] for i in perm])
#          return iter(indices)
#
#      def __len__(self) -> int:
#          return len(self.valid_indices)

def rollout(
        env: gym.vector.VectorEnv,
        policy: PreTrainedPolicy,
        batch_size: int,
        env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
        seeds: list[int] | None = None,
        return_observations: bool = False,
        return_action_context: bool = False,
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
        (optional) "prev_actions": A (batch, sequence, prev_steps, action_dim) tensor of previous actions.
        (optional) "pred_action": A (batch, sequence, pred_steps, action_dim) tensor of predicted actions.
        (optional) "actions_seq_valid": A (batch, sequence) tensor of validity flags for action context.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        return_action_context: Whether to include action context (prev_actions, pred_action, actions_seq_valid)
            in the returned rollout data. Defaults to False.
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
    
    # Action context tracking
    all_prev_actions = []
    all_pred_actions = []
    all_actions_seq_valid = []

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

        # Collect action context after select_action (before postprocessor)
        # prev_steps and pred_steps default to policy.config.attn_act_len
        if return_action_context and hasattr(policy, 'get_action_context'):
            ctx = policy.get_action_context(batch_size=batch_size)
            all_prev_actions.append(ctx['prev_actions'])
            all_pred_actions.append(ctx['pred_action'])
            all_actions_seq_valid.append(ctx['actions_seq_valid'])

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

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

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
    if return_observations:
        stacked_observations = {}
        # Only stack keys that start with "observation."
        for key in all_observations[0]:
            if key.startswith(f"{OBS_STR}."):
                stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    # Add action context to return if requested
    if return_action_context and len(all_prev_actions) > 0:
        # Filter out None values and stack
        # prev_actions and pred_action are [batch, prev_steps, action_dim] or None
        # actions_seq_valid is [batch] bool or None
        valid_prev_actions = [a for a in all_prev_actions if a is not None]
        valid_pred_actions = [a for a in all_pred_actions if a is not None]
        valid_flags = [v for v in all_actions_seq_valid if v is not None]
        
        if len(valid_prev_actions) > 0:
            # Stack along sequence dimension: [batch, sequence, prev_steps, action_dim]
            ret["prev_actions"] = torch.stack(valid_prev_actions, dim=1)
        if len(valid_pred_actions) > 0:
            ret["pred_action"] = torch.stack(valid_pred_actions, dim=1)
        if len(valid_flags) > 0:
            # Stack along sequence dimension: [batch, sequence]
            ret["actions_seq_valid"] = torch.stack(valid_flags, dim=1)

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    cmp=False,
    online=False,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        cmp: Whether to use CMP (Contrastive Model Prediction) loss. Defaults to False.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch, cmp=cmp, online=online)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def collect_episodes(
    env,
    batch_size,
    policy: PreTrainedPolicy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    start_seed: int | None = None,
    start_episode_index: int = 0,
    return_action_context: bool = False,
) -> dict:
    """Collect episodes from the environment using the current policy.

    Args:
        env: The batch of environments.
        policy: The policy to use for action selection.
        env_preprocessor: Environment-specific preprocessor.
        env_postprocessor: Environment-specific postprocessor.
        preprocessor: Policy preprocessor.
        postprocessor: Policy postprocessor.
        n_episodes: Number of episodes to collect.
        start_seed: Starting seed for environment reset.
        start_episode_index: Starting episode index for the collected episodes.
        return_action_context: Whether to collect action context (prev_actions, pred_action, 
            actions_seq_valid) for online training. Defaults to False.

    Returns:
        Dictionary containing compiled episode data ready to be added to dataset.
    """
    # Determine how many batched rollouts we need
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    all_episode_data = []
    current_episode_index = start_episode_index
    current_data_index = 0

    for batch_ix in range(n_batches):
        # Calculate how many episodes to collect in this batch
        episodes_this_batch = min(env.num_envs, n_episodes - batch_ix * env.num_envs)

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs),
                start_seed + (batch_ix * env.num_envs) + episodes_this_batch,
            )

        rollout_data = rollout(
            env=env,
            policy=policy,
            batch_size=batch_size,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=True,  # We need observations for dataset
            return_action_context=return_action_context,  # Collect action context for online training
            render_callback=None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered
        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Only process the episodes we actually need
        if episodes_this_batch < env.num_envs:
            # Slice to only include the episodes we need
            rollout_data = {
                k: v[:episodes_this_batch] if isinstance(v, torch.Tensor) and v.dim() > 0 else v
                for k, v in rollout_data.items()
            }
            done_indices = done_indices[:episodes_this_batch]

        # Compile episode data for this batch
        batch_episode_data = _compile_episode_data(
            rollout_data,
            done_indices,
            start_episode_index=current_episode_index,
            start_data_index=current_data_index,
            fps=env.unwrapped.metadata["render_fps"],
        )

        if return_action_context:
            for key in ["prev_actions", "pred_action", "actions_seq_valid"]:
                if key in rollout_data:
                    # # 诊断日志：打印 rollout_data 中的形状
                    # logging.info(f"[DEBUG collect_episodes] rollout_data['{key}'] shape: {rollout_data[key].shape}")
                    
                    # 按照与 _compile_episode_data 相同的逻辑处理每个 episode
                    ep_data_list = []
                    for ep_ix in range(rollout_data[ACTION].shape[0]):
                        if ep_ix < episodes_this_batch:
                            num_frames = done_indices[ep_ix].item() + 2
                            # 取 num_frames - 1 帧（与 ACTION 等字段一致）
                            ep_data = rollout_data[key][ep_ix, : num_frames - 1]
                            # 最后一帧复制填充
                            ep_data = torch.cat([ep_data, ep_data[-1:]])
                            
                            # 确保 actions_seq_valid 保持 (n, 1) 形状以匹配数据集定义
                            if key == "actions_seq_valid":
                                if ep_data.ndim == 1:
                                    ep_data = ep_data.unsqueeze(-1)  # [n] -> [n, 1]
                                # # 诊断日志
                                # if ep_ix == 0:
                                #     logging.info(f"[DEBUG collect_episodes] ep_data['{key}'] after unsqueeze: shape={ep_data.shape}")
                            
                            # 移到 CPU（rollout 数据可能在 CUDA 上）
                            ep_data_list.append(ep_data.cpu())
                    if ep_data_list:
                        batch_episode_data[key] = torch.cat(ep_data_list)
                        # # 诊断日志
                        # logging.info(f"[DEBUG collect_episodes] batch_episode_data['{key}'] final shape: {batch_episode_data[key].shape}")

        all_episode_data.append(batch_episode_data)
        current_episode_index += episodes_this_batch
        # Update data index for next batch
        if len(batch_episode_data.get("index", [])) > 0:
            current_data_index = batch_episode_data["index"][-1].item() + 1
    # Concatenate all episode data
    if len(all_episode_data) > 1:
        combined_data = {}
        for key in all_episode_data[0]:
            combined_data[key] = torch.cat([ep[key] for ep in all_episode_data])
        return combined_data
    elif len(all_episode_data) == 1:
        return all_episode_data[0]
    else:
        return {}


def add_episodes_to_dataset(
    online_dataset: LeRobotDataset,
    episode_data: dict,
) -> None:

    """Add collected episodes to the dataset.

    This function converts episode_data from batch format to individual frames
    and adds them to the dataset using add_frame() and save_episode().

    Args:
        online_dataset: The online dataset to add episodes to.
        episode_data: Dictionary containing episode data from collect_episodes.
            Expected keys:
            - ACTION: actions (total_frames, action_dim)
            - REWARD: rewards (total_frames,)
            - DONE: done flags (total_frames,)
            - episode_index: episode indices (total_frames,)
            - frame_index: frame indices within episodes (total_frames,)
            - timestamp: timestamps (total_frames,)
            - index: global frame indices (total_frames,)
            - observation.*: observation keys (total_frames, ...)
            - next.success: success flags (total_frames,) [optional]
            - task: task names (total_frames,) [optional, if not in observation]
    """
    import time

    # ===== Performance timing =====
    func_start_time = time.time()
    preprocess_time = 0.0
    add_frame_time = 0.0
    save_episode_time = 0.0
    image_convert_time = 0.0

    if not episode_data:
        logging.warning("episode_data is empty, nothing to add to dataset")
        return

    # Get total number of frames
    total_frames = len(episode_data[ACTION])

    # Track current episode to detect episode boundaries
    current_episode_index = None

    # Extract task information if available
    # Task might be in episode_data directly, or in observation keys
    task_key = None
    if "task" in episode_data:
        task_key = "task"
    else:
        # Check if task is in observation keys (it might be saved as "observation.task" or just "task")
        for key in episode_data:
            if key == "task" or key.endswith(".task"):
                task_key = key
                break

    # Iterate through all frames
    for frame_idx in range(total_frames):
        frame_start_time = time.time()

        # Create frame dictionary
        frame_dict = {}

        # Add action (reward and done are not in dataset features, so we don't add them)
        frame_dict[ACTION] = episode_data[ACTION][frame_idx]

        # Add task (required by add_frame)
        if task_key and task_key in episode_data:
            task_value = episode_data[task_key][frame_idx]
            # Task might be a list (from add_envs_task) or a string
            if isinstance(task_value, (list, tuple)) and len(task_value) > 0:
                frame_dict["task"] = task_value[0] if isinstance(task_value[0], str) else str(task_value[0])
            elif isinstance(task_value, torch.Tensor):
                # Convert tensor to string if needed
                task_item = task_value.item() if task_value.numel() == 1 else str(task_value)
                frame_dict["task"] = str(task_item) if not isinstance(task_item, str) else task_item
            else:
                frame_dict["task"] = str(task_value) if task_value else ""
        else:
            # Default task if not available
            frame_dict["task"] = ""

        # Don't add timestamp - it's in DEFAULT_FEATURES and will be handled by add_frame automatically

        # Add all observation keys (only those starting with "observation.")
        for key in episode_data:
            if key.startswith(f"{OBS_STR}."):
                value = episode_data[key][frame_idx]

                # Convert images from channel-first (C, H, W) to channel-last (H, W, C) if needed
                # Dataset features expect channel-last format (H, W, C) based on names=["height", "width", "channel"]
                if key in online_dataset.features:
                    feat = online_dataset.features[key]
                    if feat.get("dtype") in ["image", "video"]:
                        img_convert_start = time.time()
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().numpy()
                        if isinstance(value, np.ndarray) and value.ndim == 3:
                            # Check if it's channel-first (C, H, W) - typically C=3 is small
                            if value.shape[0] == 3 and value.shape[0] < value.shape[1] and value.shape[0] < value.shape[2]:
                                # Convert from (C, H, W) to (H, W, C)
                                value = np.transpose(value, (1, 2, 0))
                        image_convert_time += time.time() - img_convert_start

                        # Log image shape for debugging (first frame only)
                        if frame_idx == 0:
                            logging.info(
                                f"Image feature '{key}': actual_shape={value.shape if isinstance(value, np.ndarray) else type(value)}, "
                                f"expected_shape={feat.get('shape')}, names={feat.get('names')}, dtype={feat.get('dtype')}"
                            )

                frame_dict[key] = value
        # Don't add next.success as complementary_info if it's not in dataset features
        # Only add complementary_info keys if they exist in dataset features
        # if "next.success" in episode_data:
        #     complementary_info_keys = [k for k in online_dataset.features if k.startswith("complementary_info.")]
        #     if "complementary_info.success" in online_dataset.features:
        #         success_value = episode_data["next.success"][frame_idx]
        #         if isinstance(success_value, torch.Tensor):
        #             if success_value.numel() == 1:
        #                 frame_dict["complementary_info.success"] = success_value.item()
        #             else:
        #                 frame_dict["complementary_info.success"] = success_value
        #         else:
        #             frame_dict["complementary_info.success"] = success_value

        # Add action context fields if present in episode_data and dataset features
        if "prev_actions" in episode_data and "prev_actions" in online_dataset.features:
            prev_actions = episode_data["prev_actions"][frame_idx]
            if isinstance(prev_actions, torch.Tensor):
                prev_actions = prev_actions.cpu()
            frame_dict["prev_actions"] = prev_actions
        
        if "pred_action" in episode_data and "pred_action" in online_dataset.features:
            pred_action = episode_data["pred_action"][frame_idx]
            # # 诊断日志
            # prev_action = episode_data["prev_actions"][frame_idx]
            # if frame_idx == 0:
            #     logging.info(f"[DEBUG add_episodes] episode_data['pred_action'].shape={episode_data['pred_action'].shape}, "
            #                f"pred_action[0].shape={pred_action.shape if hasattr(pred_action, 'shape') else 'N/A'}")
            #     logging.info(f"[DEBUG add_episodes] episode_data['prev_actions'].shape={episode_data['prev_actions'].shape}, "
            #                f"prev_action[0].shape={prev_action.shape if hasattr(prev_action, 'shape') else 'N/A'}")
            if isinstance(pred_action, torch.Tensor):
                pred_action = pred_action.cpu()
            frame_dict["pred_action"] = pred_action
        
        if "actions_seq_valid" in episode_data and "actions_seq_valid" in online_dataset.features:
            actions_seq_valid_raw = episode_data["actions_seq_valid"][frame_idx]
            
            # # 诊断日志：打印原始数据形状
            # if frame_idx == 0:
            #     logging.info(f"[DEBUG] actions_seq_valid raw type: {type(actions_seq_valid_raw)}, "
            #                f"shape: {actions_seq_valid_raw.shape if isinstance(actions_seq_valid_raw, torch.Tensor) else 'N/A'}, "
            #                f"episode_data['actions_seq_valid'] shape: {episode_data['actions_seq_valid'].shape}")
            
            if isinstance(actions_seq_valid_raw, torch.Tensor):
                # 确保是 [1] 形状的 bool tensor，并移到 CPU
                if actions_seq_valid_raw.ndim == 1 and actions_seq_valid_raw.shape[0] == 1:
                    # 已经是 [1] 形状，确保是 bool 类型并移到 CPU
                    actions_seq_valid = actions_seq_valid_raw.cpu().bool()
                elif actions_seq_valid_raw.ndim == 0:
                    # 标量 tensor，包装成 [1] 形状
                    actions_seq_valid = torch.tensor([actions_seq_valid_raw.cpu().item()], dtype=torch.bool)
                else:
                    # 其他情况，取第一个元素
                    # logging.warning(f"[DEBUG] Unexpected actions_seq_valid shape: {actions_seq_valid_raw.shape}")
                    actions_seq_valid = torch.tensor([actions_seq_valid_raw.cpu().flatten()[0].item()], dtype=torch.bool)
            else:
                actions_seq_valid = torch.tensor([actions_seq_valid_raw], dtype=torch.bool)
            
            # # 诊断日志：打印处理后的形状
            # if frame_idx == 0:
            #     logging.info(f"[DEBUG] actions_seq_valid after processing: shape={actions_seq_valid.shape}, dtype={actions_seq_valid.dtype}")
            
            frame_dict["actions_seq_valid"] = actions_seq_valid
            
            # 不跳过帧，让所有帧都被保存到数据集
            # 训练时可以通过采样器（如 ValidActionContextSampler）来过滤无效帧
        # Log frame_dict keys and dataset features for first frame
        if frame_idx == 0:
            logging.info(f"Frame dict keys: {sorted(frame_dict.keys())}")
            logging.info(f"Dataset expected features (excluding DEFAULT_FEATURES): {sorted(set(online_dataset.features.keys()) - {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'})}")
            logging.info(f"Extra keys in frame_dict (not in dataset features): {sorted(set(frame_dict.keys()) - {'task'} - set(online_dataset.features.keys()))}")

        # Record preprocessing time for this frame
        preprocess_time += time.time() - frame_start_time

        # Check if we need to save episode before adding frame (episode boundary)
        episode_index = episode_data["episode_index"][frame_idx].item()
        # Get done flag from episode_data (not from frame_dict, as it's not in dataset features)
        done = episode_data[DONE][frame_idx] if DONE in episode_data else False

        # Save previous episode if episode index changed (new episode started)
        if current_episode_index is not None and episode_index != current_episode_index:
            if online_dataset.episode_buffer is not None and online_dataset.episode_buffer["size"] > 0:
                save_ep_start = time.time()
                online_dataset.save_episode()
                save_episode_time += time.time() - save_ep_start
                if frame_idx < 100:  # Log first few save_episode times
                    logging.info(f"[PERF] save_episode #{current_episode_index}: {time.time() - save_ep_start:.3f}s")

        # Add frame to dataset
        add_frame_start = time.time()
        online_dataset.add_frame(frame_dict)
        add_frame_time += time.time() - add_frame_start

        # Save episode if this is the last frame of the episode
        # 检测是否是 episode 的最后一帧（下一个帧属于不同的 episode，或者这是所有帧的最后一帧）
        is_last_frame_of_episode = (
            frame_idx == total_frames - 1 or
            episode_data["episode_index"][frame_idx + 1].item() != episode_index
        )

        # 如果是最后一帧，无论 done 标志如何都应该保存 episode
        # 因为 episode 已经结束了（可能是正常结束或被截断）
        if is_last_frame_of_episode:
            if online_dataset.episode_buffer is not None and online_dataset.episode_buffer["size"] > 0:
                save_ep_start = time.time()
                online_dataset.save_episode()
                save_episode_time += time.time() - save_ep_start

        # Update current episode index
        current_episode_index = episode_index
    # Save the last episode if there are any remaining frames
    # 这个检查是额外的安全措施，但通常上面的逻辑已经处理了所有情况
    if total_frames > 0:
        if online_dataset.episode_buffer is not None and online_dataset.episode_buffer["size"] > 0:
            save_ep_start = time.time()
            online_dataset.save_episode()
            save_episode_time += time.time() - save_ep_start

    # ===== Performance timing output =====
    total_time = time.time() - func_start_time
    logging.info(f"[PERF] add_episodes_to_dataset timing breakdown:")
    logging.info(f"[PERF]   Total time: {total_time:.3f}s for {total_frames} frames")
    logging.info(f"[PERF]   Preprocess time: {preprocess_time:.3f}s ({100*preprocess_time/total_time:.1f}%)")
    logging.info(f"[PERF]     - Image convert time: {image_convert_time:.3f}s ({100*image_convert_time/total_time:.1f}%)")
    logging.info(f"[PERF]   add_frame time: {add_frame_time:.3f}s ({100*add_frame_time/total_time:.1f}%)")
    logging.info(f"[PERF]   save_episode time: {save_episode_time:.3f}s ({100*save_episode_time/total_time:.1f}%)")
    logging.info(f"[PERF]   Avg time per frame: {1000*total_time/total_frames:.2f}ms")
    logging.info(f"Added {total_frames} frames to dataset")


@parser.wrap()
def online_train_main(cfg: OnlineTrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function for online training with data collection.

    This function orchestrates the online training pipeline:
    1. Sets up environment, policy, and dataset
    2. Alternates between collecting episodes and training
    3. Periodically saves checkpoints and logs metrics

    Args:
        cfg: An `OnlineTrainPipelineConfig` object containing all configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Create environment for data collection
    if is_main_process:
        logging.info("Creating environment for data collection")
    collect_env_dict = make_env(
        cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs
    )
    # Keep the full dict to use all tasks, not just the first one
    # Structure: {suite_name: {task_id: vec_env}}

    # Create or load datasets
    offline_dataset = None
    online_dataset = None
    
    if is_main_process:
        logging.info("Creating/loading datasets")
        
        # Load offline dataset if specified
        if cfg.online.start_with_offline_dataset:
            logging.info("Loading offline dataset for training")
            offline_dataset = make_dataset(cfg)
            logging.info(
                f"Offline dataset loaded: {offline_dataset.num_episodes} episodes, "
                f"{offline_dataset.num_frames} frames"
            )
        
        # Create or load online dataset for collecting new episodes
        online_dataset_repo_id = cfg.online.online_dataset_repo_id or cfg.dataset.repo_id
        online_dataset_root = cfg.online.online_dataset_root or cfg.dataset.root
        
        # Check if online dataset already exists by checking for info.json
        online_dataset_path = Path(online_dataset_root)
        online_meta_path = online_dataset_path / "meta" / "info.json"
        
        need_to_create = False
        if online_meta_path.exists():
            # Dataset exists, load it
            logging.info(f"Loading existing online dataset: {online_dataset_repo_id}")
            online_dataset = LeRobotDataset(
                online_dataset_repo_id,
                root=online_dataset_root,
                video_backend=cfg.dataset.video_backend,
            )
            # Enable async image writing for better performance (more threads = faster)
            online_dataset.start_image_writer(num_threads=cfg.num_workers)
            logging.info(
                f"Online dataset loaded: {online_dataset.num_episodes} episodes, "
                f"{online_dataset.num_frames} frames"
            )
        else:
            # Dataset doesn't exist or is incomplete
            # If directory exists but info.json doesn't, try to load first (will fail gracefully)
            # If load fails, we'll remove the directory and create a new one
            if online_dataset_path.exists():
                logging.warning(
                    f"Directory {online_dataset_path} exists but info.json is missing. "
                    f"Attempting to load dataset first..."
                )
                try:
                    online_dataset = LeRobotDataset(
                        online_dataset_repo_id,
                        root=online_dataset_root,
                        video_backend=cfg.dataset.video_backend,
                    )
                    # Enable async image writing for better performance (more threads = faster)
                    online_dataset.start_image_writer(num_threads=cfg.num_workers)
                    logging.info(
                        f"Online dataset loaded: {online_dataset.num_episodes} episodes, "
                        f"{online_dataset.num_frames} frames"
                    )
                    # Successfully loaded, don't need to create
                    need_to_create = False
                except (FileNotFoundError, NotADirectoryError, AssertionError) as e:
                    logging.warning(
                        f"Failed to load dataset from existing directory: {e}. "
                        f"Removing incomplete directory and creating new dataset."
                    )
                    import shutil
                    shutil.rmtree(online_dataset_path)
                    need_to_create = True
            else:
                # Directory doesn't exist, need to create new one
                need_to_create = True
            
            if need_to_create:
                logging.info(f"Creating new online dataset: {online_dataset_repo_id}")
                
                # Get configuration from offline dataset if available, otherwise from environment
                if offline_dataset is not None:
                    # Copy configuration from offline dataset
                    fps = offline_dataset.meta.fps
                    features = offline_dataset.meta.features.copy()
                    robot_type = offline_dataset.meta.robot_type
                    logging.info(
                        f"Using configuration from offline dataset: fps={fps}, "
                    )
                else:
                    # Get configuration from environment
                    fps = cfg.env.fps
                    # Convert environment features to dataset features format
                    from lerobot.envs.utils import env_to_policy_features
                    
                    # Get policy features from env config
                    policy_features = env_to_policy_features(cfg.env)
                    
                    # Convert to dataset features format
                    features = {}
                    for key, feat in policy_features.items():
                        if feat.type.name == "VISUAL":
                            # For visual features, use video dtype and keep channel-last shape
                            # Note: env features are already in channel-last format (h, w, c)
                            features[key] = {
                                "dtype": "video",
                                "shape": feat.shape,  # (h, w, c)
                                "names": ["height", "width", "channels"],
                            }
                        elif feat.type.name == "STATE" or feat.type.name == "ACTION":
                            features[key] = {
                                "dtype": "float32",
                                "shape": feat.shape,
                                "names": None,
                            }
                        else:
                            features[key] = {
                                "dtype": "float32",
                                "shape": feat.shape,
                                "names": None,
                            }
                    
                    # Add default features (reward, done)
                    features[REWARD] = {"dtype": "float32", "shape": (1,), "names": None}
                    features[DONE] = {"dtype": "bool", "shape": (1,), "names": None}
                    
                    robot_type = None
                    logging.info(
                        f"Using configuration from environment: fps={fps}, "
                    )
                
                # Add action context features for online training
                # Get action dimension from features
                action_dim = features[ACTION]["shape"][-1] if ACTION in features else 7  # default to 7
                # Get attn_act_len from policy config (defaults to 10 for backward compatibility)
                attn_act_len = getattr(cfg.policy, 'attn_act_len', 10)
                prev_steps = attn_act_len
                pred_steps = attn_act_len
                
                features["prev_actions"] = {
                    "dtype": "float32",
                    "shape": (prev_steps, action_dim),
                    "names": ["prev_step", "action_dim"],
                }
                features["pred_action"] = {
                    "dtype": "float32",
                    "shape": (pred_steps, action_dim),
                    "names": ["pred_step", "action_dim"],
                }
                features["actions_seq_valid"] = {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": None,
                }
                logging.info(
                    f"Added action context features: prev_actions shape={(prev_steps, action_dim)}, "
                    f"pred_action shape={(pred_steps, action_dim)}"
                )
                
                # Create empty online dataset
                online_dataset = LeRobotDataset.create(
                    online_dataset_repo_id,
                    fps=fps,
                    features=features,
                    root=online_dataset_root,
                    robot_type=robot_type,
                    batch_encoding_size=cfg.dataset.video_encoding_batch_size if hasattr(cfg.dataset, "video_encoding_batch_size") else 1,
                    image_writer_threads=16,  # Enable async image writing for better performance
                )
                logging.info("Empty online dataset created successfully")

    accelerator.wait_for_everyone()

    if not is_main_process:
        # Load datasets on non-main processes
        if cfg.online.start_with_offline_dataset:
            offline_dataset = make_dataset(cfg)
        
        online_dataset_repo_id = cfg.online.online_dataset_repo_id or cfg.dataset.repo_id
        online_dataset_root = cfg.online.online_dataset_root or cfg.dataset.root
        try:
            online_dataset = LeRobotDataset(
                online_dataset_repo_id,
                root=online_dataset_root,
                video_backend=cfg.dataset.video_backend,
            )
        except (FileNotFoundError, NotADirectoryError):
            # This shouldn't happen if main process created it, but handle gracefully
            raise RuntimeError(
                f"Online dataset {online_dataset_repo_id} not found. "
                "It should have been created by the main process."
            )
    
    # Use offline dataset for training if available, otherwise use online dataset
    # The online dataset will be used for collecting new episodes
    training_dataset = offline_dataset if offline_dataset is not None else online_dataset

    # Create policy
    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=training_dataset.meta,
        rename_map=cfg.rename_map,
    )

    accelerator.wait_for_everyone()

    # Create processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = training_dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": training_dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": training_dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # Create environment-specific processors
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env)

    # Create separate preprocessor for data collection (without normalizer)
    # This matches lerobot_eval.py's approach - we want raw data, not normalized data
    collect_preprocessor_overrides = {
        "device_processor": {"device": device.type},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    collect_preprocessor, collect_postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=collect_preprocessor_overrides,
    )

    # Log processor steps for debugging
    if is_main_process:
        from lerobot.processor.normalize_processor import NormalizerProcessorStep
        processor_names = [type(step).__name__ for step in collect_preprocessor.steps]
        has_normalizer = any(isinstance(step, NormalizerProcessorStep) for step in collect_preprocessor.steps)
        logging.info(f"Collect preprocessor steps: {processor_names}")
        logging.info(f"Collect preprocessor contains normalizer_processor: {has_normalizer}")
        
        # Also log training preprocessor for comparison
        training_processor_names = [type(step).__name__ for step in preprocessor.steps]
        training_has_normalizer = any(isinstance(step, NormalizerProcessorStep) for step in preprocessor.steps)
        logging.info(f"Training preprocessor steps: {training_processor_names}")
        logging.info(f"Training preprocessor contains normalizer_processor: {training_has_normalizer}")

    # Create FillMissingActionContextProcessor for offline dataset training
    # Get action dimension from training dataset
    action_dim = training_dataset.meta.features[ACTION]["shape"][-1]
    # Get attn_act_len from policy config (defaults to 10 for backward compatibility)
    attn_act_len = getattr(cfg.policy, 'attn_act_len', 10)
    fill_action_context_processor = FillMissingActionContextProcessor(
        action_dim=action_dim,
        prev_steps=attn_act_len,
        pred_steps=attn_act_len,
    )
    if is_main_process:
        logging.info(f"Created FillMissingActionContextProcessor with action_dim={action_dim}, attn_act_len={attn_act_len}")

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # Global training step counter

    if cfg.resume:
        from lerobot.utils.train_utils import load_training_state

        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.online.n_iterations=}")
        logging.info(f"{cfg.online.collect_episodes_per_iteration=}")
        logging.info(f"{cfg.online.train_steps_per_iteration=}")
        if offline_dataset is not None:
            logging.info(
                f"Offline dataset: {offline_dataset.num_episodes} episodes, "
                f"{offline_dataset.num_frames} frames ({format_big_number(offline_dataset.num_frames)})"
            )
        logging.info(
            f"Online dataset: {online_dataset.num_episodes} episodes, "
            f"{online_dataset.num_frames} frames ({format_big_number(online_dataset.num_frames)})"
        )
        logging.info(
            f"Training dataset: {training_dataset.num_episodes} episodes, "
            f"{training_dataset.num_frames} frames ({format_big_number(training_dataset.num_frames)})"
        )
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Create dataloaders for offline and online datasets
    def create_dataloader(dataset, online=False):
        """创建 DataLoader。
        
        Args:
            dataset: 数据集
            use_valid_action_sampler: 是否使用 ValidActionContextSampler
                仅采样 actions_seq_valid=True 的帧（用于在线数据集）
        """

        if online:
            shuffle = False
            sampler = ValidActionContextSampler(
                dataset,
                shuffle=True,
            )
        elif hasattr(cfg.policy, "drop_n_last_frames"):
            shuffle = False
            sampler = EpisodeAwareSampler(
                dataset.meta.episodes["dataset_from_index"],
                dataset.meta.episodes["dataset_to_index"],
                episode_indices_to_use=dataset.episodes,
                drop_n_last_frames=cfg.policy.drop_n_last_frames,
                shuffle=True,
            )
        else:
            shuffle = True
            sampler = None

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=shuffle and not cfg.dataset.streaming,
            sampler=sampler,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )

    # Create dataloader for offline dataset (if exists)
    offline_dataloader = None
    offline_dl_iter = None
    if offline_dataset is not None:
        offline_dataloader = create_dataloader(offline_dataset)
        if is_main_process:
            logging.info("Created dataloader for offline dataset")

    # Create dataloader for online dataset (only if it has data)
    # Use ValidActionContextSampler to only sample valid frames
    online_dataloader = None
    online_dl_iter = None
    online_sampler = None  # 保存 sampler 引用以便后续刷新
    if online_dataset.num_frames > 0:
        online_dataloader = create_dataloader(online_dataset, online=True)
        # 保存 sampler 引用
        online_sampler = online_dataloader.sampler if hasattr(online_dataloader, 'sampler') else None
        if is_main_process:
            valid_count = len(online_sampler) if online_sampler else online_dataset.num_frames
            logging.info(
                f"Created dataloader for online dataset with ValidActionContextSampler "
                f"({valid_count} valid frames)"
            )
    else:
        if is_main_process:
            logging.info("Online dataset is empty, dataloader will be created after first data collection")

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    prepare_list = [policy, optimizer, lr_scheduler]
    if offline_dataloader is not None:
        prepare_list.append(offline_dataloader)
    if online_dataloader is not None:
        prepare_list.append(online_dataloader)
    
    prepared = accelerator.prepare(*prepare_list)
    if offline_dataloader is not None and online_dataloader is not None:
        policy, optimizer, lr_scheduler, offline_dataloader, online_dataloader = prepared
        offline_dl_iter = cycle(offline_dataloader)
        online_dl_iter = cycle(online_dataloader)
    elif offline_dataloader is not None:
        policy, optimizer, lr_scheduler, offline_dataloader = prepared
        offline_dl_iter = cycle(offline_dataloader)
    elif online_dataloader is not None:
        policy, optimizer, lr_scheduler, online_dataloader = prepared
        online_dl_iter = cycle(online_dataloader)
    else:
        policy, optimizer, lr_scheduler = prepared

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        training_dataset.num_frames,
        training_dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        if cfg.online.online_collect:
            logging.info("Start online training: alternating between data collection and training")
        else:
            logging.info("Start training on existing online dataset (no new data collection)")

    # Main online training loop
    for iteration in range(cfg.online.n_iterations):
        if is_main_process:
            logging.info(f"\n{'='*60}")
            logging.info(f"Iteration {iteration + 1}/{cfg.online.n_iterations}")
            logging.info(f"{'='*60}")

        # Phase 1: Collect episodes from all tasks (only if online=True)
        if cfg.online.online_collect:
            import time as _time
            phase1_start = _time.time()

            if is_main_process:
                logging.info(
                    f"Collecting {cfg.online.collect_episodes_per_iteration} episodes from environment"
                )
            policy.eval()
            with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
                # Get current episode count from online dataset to set correct episode indices
                current_episode_count = online_dataset.num_episodes if hasattr(online_dataset, "num_episodes") else 0
                
                # Collect episodes from all tasks
                all_episode_data = []
                total_tasks = sum(len(tasks) for tasks in collect_env_dict.values())
                episodes_per_task = cfg.online.collect_episodes_per_iteration // total_tasks
                remaining_episodes = cfg.online.collect_episodes_per_iteration % total_tasks
                
                current_ep_idx = current_episode_count
                task_idx = 0
                
                for suite_name, task_dict in collect_env_dict.items():
                    for task_id, env in task_dict.items():
                        # Distribute remaining episodes to first few tasks
                        n_episodes_this_task = episodes_per_task + (1 if task_idx < remaining_episodes else 0)
                        
                        if n_episodes_this_task > 0:
                            if is_main_process:
                                logging.info(
                                    f"Collecting {n_episodes_this_task} episodes from {suite_name} task {task_id}"
                                )
                            
                            task_episode_data = collect_episodes(
                                env=env,
                                batch_size=cfg.eval.batch_size,
                                policy=accelerator.unwrap_model(policy),
                                env_preprocessor=env_preprocessor,
                                env_postprocessor=env_postprocessor,
                                preprocessor=collect_preprocessor,  # Use collect_preprocessor (without normalizer)
                                postprocessor=collect_postprocessor,  # Use collect_postprocessor (without unnormalizer)
                                n_episodes=n_episodes_this_task,
                                start_seed=cfg.seed if cfg.seed is not None else None,
                                start_episode_index=current_ep_idx,
                                return_action_context=True,  # Collect action context for online training
                            )
                            
                            if task_episode_data:
                                all_episode_data.append(task_episode_data)
                                # Update episode index for next task
                                # Count unique episode indices to get the number of episodes collected
                                if "episode_index" in task_episode_data and len(task_episode_data["episode_index"]) > 0:
                                    unique_episodes = torch.unique(task_episode_data["episode_index"])
                                    current_ep_idx = unique_episodes[-1].item() + 1
                                else:
                                    # Fallback: increment by number of episodes collected
                                    current_ep_idx += n_episodes_this_task
                        
                        task_idx += 1
                
                # Combine all episode data
                if len(all_episode_data) > 1:
                    episode_data = {}
                    for key in all_episode_data[0]:
                        episode_data[key] = torch.cat([ep[key] for ep in all_episode_data])
                elif len(all_episode_data) == 1:
                    episode_data = all_episode_data[0]
                else:
                    episode_data = {}

            phase1_time = _time.time() - phase1_start
            if is_main_process:
                logging.info(f"[PERF] Phase 1 (collect_episodes): {phase1_time:.3f}s")

            # Phase 2: Add episodes to online dataset
            phase2_start = _time.time()
            if is_main_process:
                logging.info("Adding collected episodes to online dataset")
                add_episodes_to_dataset(online_dataset, episode_data)
            phase2_time = _time.time() - phase2_start
            if is_main_process:
                logging.info(f"[PERF] Phase 2 (add_episodes_to_dataset): {phase2_time:.3f}s")
                logging.info(f"[PERF] Total data collection + dataset update: {phase1_time + phase2_time:.3f}s")

            # Close all writers to ensure parquet files are properly finalized before reading
            if is_main_process:
                online_dataset._close_writer()
                if hasattr(online_dataset.meta, "_close_writer"):
                    online_dataset.meta._close_writer()
                online_dataset._ensure_hf_dataset_loaded()

            # Wait for dataset update to complete
            accelerator.wait_for_everyone()

            # Reload online dataset to include new episodes
            # Note: The online_dataset should be updated in-place by add_episodes_to_dataset
            # If the dataset implementation requires reloading, do it here
            # Force reload dataset metadata if needed
            if hasattr(online_dataset.meta, "reload"):
                online_dataset.meta.reload()
            
            # Recreate dataloaders with updated datasets
            if offline_dataset is not None:
                offline_dataloader = create_dataloader(offline_dataset)
                offline_dataloader = accelerator.prepare(offline_dataloader)
                offline_dl_iter = cycle(offline_dataloader)
            
            # Only create online dataloader if dataset has data
            # Use ValidActionContextSampler to only sample valid frames
            if online_dataset.num_frames > 0:
                online_dataloader = create_dataloader(online_dataset, online=True)
                # 保存 sampler 引用
                online_sampler = online_dataloader.sampler if hasattr(online_dataloader, 'sampler') else None
                online_dataloader = accelerator.prepare(online_dataloader)
                online_dl_iter = cycle(online_dataloader)
                if is_main_process:
                    valid_count = len(online_sampler) if online_sampler else online_dataset.num_frames
                    logging.info(
                        f"Recreated online dataloader with ValidActionContextSampler "
                        f"({valid_count} valid frames out of {online_dataset.num_frames})"
                    )
            else:
                if is_main_process:
                    logging.warning("Online dataset is still empty, skipping dataloader creation")

        # Check if we have enough episodes to start training
        # Count total episodes: offline + online
        total_episodes = (
            (offline_dataset.num_episodes if offline_dataset is not None else 0)
            + online_dataset.num_episodes
        )
        if total_episodes < cfg.online.min_episodes_for_training:
            if is_main_process:
                logging.info(
                    f"Only {total_episodes} total episodes (offline: "
                    f"{offline_dataset.num_episodes if offline_dataset is not None else 0}, "
                    f"online: {online_dataset.num_episodes}), "
                    f"need {cfg.online.min_episodes_for_training}. Skipping training phase."
                )
            continue

        # Phase 3: Train on both offline and online datasets
        # Each training step consists of two sub-steps:
        # 1. Train on offline dataset with cmp=False
        # 2. Train on online dataset with cmp=True
        if is_main_process:
            offline_info = (
                f"{offline_dataset.num_episodes} episodes from offline dataset"
                if offline_dataset is not None
                else "no offline dataset"
            )
            online_info = f"{online_dataset.num_episodes} episodes from online dataset"
            logging.info(
                f"Training for {cfg.online.train_steps_per_iteration} steps "
                f"({offline_info}, {online_info})"
            )

        policy.train()
        iteration_start_step = step
        for train_step in range(cfg.online.train_steps_per_iteration):
            if offline_dataset is not None and offline_dataset.num_episodes > 0:
                start_time = time.perf_counter()
                offline_batch = next(offline_dl_iter)
                offline_batch = preprocessor(offline_batch)
                # Fill missing action context fields for offline dataset
                offline_batch = fill_action_context_processor(offline_batch)
                train_tracker.dataloading_s = time.perf_counter() - start_time

                train_tracker, output_dict = update_policy(
                    train_tracker,
                    policy,
                    offline_batch,
                    optimizer,
                    cfg.optimizer.grad_clip_norm,
                    accelerator=accelerator,
                    lr_scheduler=lr_scheduler,
                    cmp=(train_step%4==0),
                )

                step += 1
                train_tracker.step()


            if (online_dataset.num_episodes > 0 and online_dataloader is not None
                    and online_dataset.num_frames >= cfg.online.min_frames_for_online_training):
                start_time = time.perf_counter()
                online_batch = next(online_dl_iter)
                online_batch = preprocessor(online_batch)
                train_tracker.dataloading_s = time.perf_counter() - start_time

                train_tracker, output_dict = update_policy(
                    train_tracker,
                    policy,
                    online_batch,
                    optimizer,
                    cfg.optimizer.grad_clip_norm,
                    accelerator=accelerator,
                    lr_scheduler=lr_scheduler,
                    online=True,  # Train online data with cmp=True
                )

                step += 1
                train_tracker.step()


            # Log and save checkpoints after each training iteration
            # (which includes both offline and online training steps)
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
            is_saving_step = step % cfg.save_freq == 0

            if is_log_step:
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_log_dict["iteration"] = iteration + 1
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and is_saving_step:
                if is_main_process:
                    logging.info(f"Checkpoint policy after step {step}")
                    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=step,
                        cfg=cfg,
                        policy=accelerator.unwrap_model(policy),
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                    )
                    update_last_checkpoint(checkpoint_dir)
                    if wandb_logger:
                        wandb_logger.log_policy(checkpoint_dir)

                accelerator.wait_for_everyone()

        if is_main_process:
            total_episodes = (
                (offline_dataset.num_episodes if offline_dataset is not None else 0)
                + online_dataset.num_episodes
            )
            total_frames = (
                (offline_dataset.num_frames if offline_dataset is not None else 0)
                + online_dataset.num_frames
            )
            if cfg.online.online_collect:
                logging.info(
                    f"Iteration {iteration + 1} complete: "
                    f"collected {cfg.online.collect_episodes_per_iteration} episodes, "
                    f"trained {step - iteration_start_step} steps. "
                    f"Total: {total_episodes} episodes ({offline_dataset.num_episodes if offline_dataset is not None else 0} offline + {online_dataset.num_episodes} online), "
                    f"{total_frames} frames"
                )
            else:
                logging.info(
                    f"Iteration {iteration + 1} complete: "
                    f"trained {step - iteration_start_step} steps (no new data collected). "
                    f"Total: {total_episodes} episodes ({offline_dataset.num_episodes if offline_dataset is not None else 0} offline + {online_dataset.num_episodes} online), "
                    f"{total_frames} frames"
                )

    # Final checkpoint
    if cfg.save_checkpoint and is_main_process:
        logging.info(f"Final checkpoint at step {step}")
        checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=step,
            cfg=cfg,
            policy=accelerator.unwrap_model(policy),
            optimizer=optimizer,
            scheduler=lr_scheduler,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        update_last_checkpoint(checkpoint_dir)
        if wandb_logger:
            wandb_logger.log_policy(checkpoint_dir)

    if collect_env_dict:
        close_envs(collect_env_dict)

    if is_main_process:
        logging.info("End of online training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    init_logging()
    online_train_main()


if __name__ == "__main__":
    main()

