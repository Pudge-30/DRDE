#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import builtins
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    GemmaForCausalLM = None
    PaliGemmaForConditionalGeneration = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
        time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
        images: torch.Tensor,
        height: int,
        width: int,
        mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


def _adjust_attention_and_position_for_none_inputs(
        inputs_embeds: list[torch.Tensor | None],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjust attention_mask and position_ids when inputs_embeds contains None.

    Args:
        inputs_embeds: List of input embeddings, may contain None
        attention_mask: Original attention mask [batch, seq_len, seq_len] or [batch, num_heads, seq_len, seq_len]
        position_ids: Original position ids [batch, seq_len]

    Returns:
        Adjusted attention_mask and position_ids
    """
    if inputs_embeds[1] is None and inputs_embeds[0] is not None:
        # Only prefix exists, need to extract prefix part
        prefix_len = inputs_embeds[0].shape[1]
        # Extract prefix part from attention_mask
        if attention_mask.dim() == 4:
            # [batch, num_heads, seq_len, seq_len]
            attention_mask = attention_mask[:, :, :prefix_len, :prefix_len]
        else:
            # [batch, seq_len, seq_len]
            attention_mask = attention_mask[:, :prefix_len, :prefix_len]
        # Extract prefix part from position_ids
        position_ids = position_ids[:, :prefix_len]
    # If both are not None, no adjustment needed

    return attention_mask, position_ids


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.language_model, gemma_expert.model]
    # Track valid (non-None) inputs and their indices
    valid_indices = []
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        if hidden_states is None:
            continue
        valid_indices.append(i)
        layer = models[i].layers[layer_idx]
        cond = adarms_cond[i] if adarms_cond is not None else None
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=cond)
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    # If no valid inputs, return None for all outputs
    if not valid_indices:
        return [None] * len(inputs_embeds)

    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    valid_idx = 0
    for i, hidden_states in enumerate(inputs_embeds):
        if hidden_states is None:
            outputs_embeds.append(None)
            continue
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[valid_idx])  # noqa: SLF001
        after_first_residual = out_emb.clone()
        cond = adarms_cond[i] if adarms_cond is not None else None
        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=cond)
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
        outputs_embeds.append(out_emb)
        start_pos = end_pos
        valid_idx += 1
    return outputs_embeds


# def vicreg_loss(
#         z1: Tensor,
#         z2: Tensor,
#         lambda_param: float = 25.0,
#         mu_param: float = 25.0,
#         nu_param: float = 1.0,
#         gamma: float = 0.4,
#         eps: float = 1e-4,
# ) -> Tensor:
#     """VICReg loss with Variance-Invariance-Covariance Regularization (PyTorch version).

#     Args:
#         z1: First representation [batch, num_tokens, dim] or [batch, dim]
#         z2: Second representation [batch, num_tokens, dim] or [batch, dim]
#         lambda_param: Weight for invariance loss (default: 25.0)
#         mu_param: Weight for variance loss (default: 25.0)
#         nu_param: Weight for covariance loss (default: 1.0)
#         gamma: Target standard deviation (default: 1.0)
#         eps: Small constant for numerical stability

#     Returns:
#         VICReg loss value (scalar tensor)
#     """
#     # Handle both 2D and 3D inputs
#     if z1.dim() == 2:
#         z1 = z1.unsqueeze(1)  # [batch, dim] -> [batch, 1, dim]
#     if z2.dim() == 2:
#         z2 = z2.unsqueeze(1)  # [batch, dim] -> [batch, 1, dim]

#     batch_size, num_tokens, dim = z1.shape

#     # Reshape to (batch*num_tokens, dim)
#     z1_flat = z1.reshape(-1, dim)  # [batch*num_tokens, dim]
#     z2_flat = z2.reshape(-1, dim)  # [batch*num_tokens, dim]
#     n_samples = z1_flat.shape[0]

#     # Invariance loss: L2 distance between corresponding representations
#     invariance_loss = torch.mean(torch.square(z1_flat - z2_flat), dim=-1)  # [batch*num_tokens]
#     invariance_loss = torch.mean(invariance_loss)  # scalar

#     # Variance loss: encourage standard deviation to be close to gamma
#     std_z1 = torch.sqrt(torch.var(z1_flat, dim=0) + eps)  # [dim]
#     std_z2 = torch.sqrt(torch.var(z2_flat, dim=0) + eps)  # [dim]
#     variance_loss = torch.mean(F.relu(gamma - std_z1)) + torch.mean(F.relu(gamma - std_z2))

#     # print(std_z1)
#     # print(std_z2)
#     # print("++++++++++")
#     # Covariance loss: encourage decorrelation of features
#     z1_centered = z1_flat - torch.mean(z1_flat, dim=0, keepdim=True)  # [batch*num_tokens, dim]
#     z2_centered = z2_flat - torch.mean(z2_flat, dim=0, keepdim=True)  # [batch*num_tokens, dim]

#     cov_z1 = (z1_centered.T @ z1_centered) / (n_samples - 1)  # [dim, dim]
#     cov_z2 = (z2_centered.T @ z2_centered) / (n_samples - 1)  # [dim, dim]

#     # Off-diagonal mask
#     off_diagonal_mask = 1 - torch.eye(dim, device=z1.device, dtype=z1.dtype)
#     offdiag_z1 = cov_z1 * off_diagonal_mask
#     offdiag_z2 = cov_z2 * off_diagonal_mask

#     # Covariance loss (normalize by dim, not number of elements)
#     cov_loss_z1 = torch.sum(torch.square(offdiag_z1)) / (dim*dim)
#     cov_loss_z2 = torch.sum(torch.square(offdiag_z2)) / (dim*dim)
#     covariance_loss = cov_loss_z1 + cov_loss_z2

#     # Total loss
#     total_loss = lambda_param * invariance_loss + mu_param * variance_loss + nu_param * covariance_loss

#     # print("----------------")
#     # print(lambda_param * invariance_loss)
#     # print(mu_param * variance_loss)
#     # print(nu_param * covariance_loss)

#     return total_loss


# def contrastive_triplet_loss(
#         z1: Tensor,
#         z2: Tensor,
#         lambda_param: float = 1.0,
#         mu_param: float = 0.0,
#         nu_param: float = 0.0,
#         gamma: float = 0.5,
#         eps: float = 1e-4,
#         temperature: float = 0.07,
#         use_triplet: bool = True,
# ) -> Tensor:
#     """基于正负样本对的对比学习损失函数，支持 Triplet Loss 和 InfoNCE 风格。
#
#     该函数使用 z1 和 z2 作为正样本对（应该相似），并从 batch 内构造负样本。
#     支持两种模式：
#     1. Triplet Loss: 使用 margin 控制正负样本之间的距离
#     2. InfoNCE 风格: 使用温度参数和 softmax 进行对比学习
#
#     Args:
#         z1: First representation [batch, num_tokens, dim] or [batch, dim]
#         z2: Second representation [batch, num_tokens, dim] or [batch, dim]
#         lambda_param: 正样本对损失权重（Triplet Loss 中的 margin，或 InfoNCE 中的主损失权重）
#         mu_param: 负样本损失权重（可选，用于平衡正负样本）
#         nu_param: 正则化项权重（可选，用于防止表示坍塌）
#         gamma: Triplet Loss 的 margin，或 InfoNCE 的温度参数（当 use_triplet=False 时）
#         eps: 数值稳定性常数
#         temperature: InfoNCE 的温度参数（当 use_triplet=False 时使用）
#         use_triplet: 是否使用 Triplet Loss（True）或 InfoNCE 风格（False）
#
#     Returns:
#         对比学习损失值（标量张量）
#     """
#     # Handle both 2D and 3D inputs
#     if z1.dim() == 2:
#         z1 = z1.unsqueeze(1)  # [batch, dim] -> [batch, 1, dim]
#     if z2.dim() == 2:
#         z2 = z2.unsqueeze(1)  # [batch, dim] -> [batch, 1, dim]
#
#     batch_size, num_tokens, dim = z1.shape
#
#     # Reshape to (batch*num_tokens, dim)
#     z1_flat = z1.reshape(-1, dim)  # [batch*num_tokens, dim]
#     z2_flat = z2.reshape(-1, dim)  # [batch*num_tokens, dim]
#     n_samples = z1_flat.shape[0]
#
#     # Normalize embeddings for stable distance computation
#     z1_norm = F.normalize(z1_flat, p=2, dim=-1)  # [batch*num_tokens, dim]
#     z2_norm = F.normalize(z2_flat, p=2, dim=-1)  # [batch*num_tokens, dim]
#
#     if use_triplet:
#         # Triplet Loss 模式
#         # z1 作为 anchor，z2 作为 positive
#         # 从 batch 内其他样本构造 negative
#
#         # 计算正样本对距离（anchor 和 positive）
#         pos_dist = torch.sum(torch.square(z1_norm - z2_norm), dim=-1)  # [n_samples]
#
#         # 构造负样本：使用 batch 内其他样本作为 negative
#         # 对于每个样本 i，z1[i] 和 z2[i] 是正样本对
#         # z1[i] 和 z2[j] (j != i) 是负样本对
#
#         # 计算 z1 与所有 z2 的成对距离
#         # z1_norm: [n_samples, dim], z2_norm: [n_samples, dim]
#         # 计算所有成对距离: [n_samples, n_samples]
#         all_distances = torch.cdist(z1_norm, z2_norm, p=2) ** 2  # [n_samples, n_samples]
#
#         # 创建掩码，排除对角线（正样本对）
#         eye_mask = torch.eye(n_samples, dtype=torch.bool, device=z1.device)
#
#         # 对于每个样本，找到最近的负样本（hard negative mining）
#         # 将正样本对的距离设为很大的值，这样 min 就不会选到它们
#         neg_distances = all_distances.clone()
#         neg_distances[eye_mask] = float('inf')
#         neg_dist = torch.min(neg_distances, dim=-1)[0]  # [n_samples] - 使用最近的负样本
#
#         # Triplet Loss: max(0, pos_dist - neg_dist + margin)
#         # margin 由 gamma 参数控制
#         triplet_loss = F.relu(pos_dist - neg_dist + gamma)  # [n_samples]
#         triplet_loss = torch.mean(triplet_loss)  # scalar
#
#         # 可选的正样本对拉近损失（鼓励正样本对更相似）
#         pos_loss = torch.mean(pos_dist)  # scalar
#
#         # 总损失
#         total_loss = lambda_param * triplet_loss + mu_param * pos_loss
#
#     else:
#         # InfoNCE 风格对比学习
#         # z1 和 z2 作为正样本对，batch 内其他样本作为负样本
#
#         # 计算相似度矩阵（使用余弦相似度）
#         # z1_norm: [n_samples, dim], z2_norm: [n_samples, dim]
#         similarity_matrix = torch.matmul(z1_norm, z2_norm.T)  # [n_samples, n_samples]
#
#         # 对角线是正样本对的相似度
#         pos_similarity = torch.diag(similarity_matrix)  # [n_samples]
#
#         # 应用温度参数
#         pos_similarity = pos_similarity / temperature  # [n_samples]
#
#         # 对于每个样本，所有其他样本都是负样本
#         # 计算 softmax 分母（包含正样本和所有负样本）
#         # 对每一行应用 softmax
#         logits = similarity_matrix / temperature  # [n_samples, n_samples]
#
#         # InfoNCE Loss: -log(exp(pos_sim) / sum(exp(all_sim)))
#         # 即：-pos_sim + log(sum(exp(all_sim)))
#         log_sum_exp = torch.logsumexp(logits, dim=-1)  # [n_samples]
#         info_nce_loss = -pos_similarity + log_sum_exp  # [n_samples]
#         info_nce_loss = torch.mean(info_nce_loss)  # scalar
#
#         # 可选的正样本对拉近损失
#         pos_loss = torch.mean(1.0 - pos_similarity * temperature)  # 转换为距离形式
#
#         # 总损失
#         total_loss = lambda_param * info_nce_loss + mu_param * pos_loss
#
#     # 可选的正则化项（防止表示坍塌）
#     if nu_param > 0:
#         # 鼓励表示的方差不为零
#         std_z1 = torch.sqrt(torch.var(z1_flat, dim=0) + eps)  # [dim]
#         std_z2 = torch.sqrt(torch.var(z2_flat, dim=0) + eps)  # [dim]
#         reg_loss = torch.mean(F.relu(gamma - std_z1)) + torch.mean(F.relu(gamma - std_z2))
#         total_loss = total_loss + nu_param * reg_loss
#
#     return total_loss


class ITMHead(nn.Module):
    """Image-Text Matching Head (ALBEF style).

    对两个独立编码的特征做二分类匹配判断，不依赖 dot product。
    参考: ALBEF (Li et al., NeurIPS 2021) 的 ITM loss。
    """

    def __init__(self, hidden_dim: int = 512, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        # 融合两个特征后做二分类
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, proj_dim),  # concat(z1, z2, z1-z2, z1*z2)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),  # 匹配分数 (logit)
        )
        self._init_weights()
        self._log_counter = 0  # Debug logging counter

    def _init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """计算匹配分数。

        Args:
            z1: [N, hidden_dim]
            z2: [N, hidden_dim]

        Returns:
            logits: [N] 匹配分数（未经 sigmoid）
        """
        combined = torch.cat([z1, z2, z1 - z2, z1 * z2], dim=-1)  # [N, 4*hidden]
        return self.classifier(combined).squeeze(-1)  # [N]

    def compute_loss(
            self,
            z1: Tensor,
            z2: Tensor,
            action_distances: Tensor = None,
            soft_label_beta: float = 1.0,
            neg_z2: Tensor = None,
            neg_z2_valid: Tensor = None,
            neg_distance: Tensor = None,  # 新增：neg 距离（用于 soft label）
    ) -> tuple[Tensor, dict]:
        """计算 ITM loss。

        使用同 episode GT 时序错位负样本 + action-distance soft label。

        Args:
            z1: [B, hidden_dim]
            z2: [B, hidden_dim]
            action_distances: [B, B] action 空间的 pairwise 距离矩阵，None 时退化为硬标签
            soft_label_beta: 软标签温度，越小越接近硬标签
            neg_z2: [B, hidden_dim] 另一侧的负样本编码
                    - L1 (obs↔action): 同 episode GT 时序错位的 action 编码
                    - L2 (action↔future_obs): 同 episode 远离 t+k 的 obs 编码
            neg_z2_valid: [B] bool，标记每个样本的 neg_z2 是否有效
            neg_distance: [B] float，neg 距离（L1: action L2 距离，L2: obs 嵌入 L2 距离）

        Returns:
            loss: scalar
            info: 诊断信息 dict
        """
        # 功能总结：基于正样本与时序错位负样本的二分类匹配损失，
        #         支持软标签（由距离决定）并记录诊断信息。
        batch_size = z1.shape[0]  # batch 大小
        device = z1.device  # 当前设备

        # 正样本：(z1_i, z2_i)
        pos_logits = self.forward(z1, z2)  # [B] 正样本匹配分数
        pos_labels = torch.ones(batch_size, device=device)  # 正样本标签=1

        if neg_z2 is not None:
            # ===== GT 时序错位负样本（替换 batch cross-pair）=====
            # 使用 action-distance soft label：距离近 → 不推太远，距离远 → 推远
            neg_logits = self.forward(z1, neg_z2)  # [B] 负样本匹配分数

            # 计算 soft label
            if neg_distance is not None:
                with torch.no_grad():
                    # 计算 scale：使用 action_distances 的 median，或 fallback 到固定值
                    if action_distances is not None:  # L1 使用 action 距离做尺度
                        eps = 1e-6
                        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)  # 对角线 mask
                        non_diag = action_distances[~eye_mask]  # 去除自距离
                        scale = non_diag.median().detach() + eps  # 使用 median 作为尺度
                    else:
                        scale = 1.0  # L2 (obs) 没有 action_distances，用固定 scale

                    # soft_label = exp(-distance / (scale × beta))
                    # distance 小 → label 接近 1（不推远），distance 大 → label 接近 0（推远）
                    neg_labels = torch.exp(-neg_distance / (scale * soft_label_beta))  # 软负标签，根据负样本距离生成一对距离敏感的标签，用于计算负样本的BCE

                    # Debug logging (每 100 次调用打印一次)
                    self._log_counter += 1  # 调试计数器累加
                    if self._log_counter % 100 == 0:
                        label_type = "L1" if action_distances is not None else "L2"
                        print(f"[ITMHead {label_type}] call={self._log_counter}, "
                              f"scale={scale:.4f}, beta={soft_label_beta:.2f}")
                        print(f"  neg_distance: mean={neg_distance.mean().item():.4f}, "
                              f"min={neg_distance.min().item():.4f}, "
                              f"max={neg_distance.max().item():.4f}")
                        print(f"  soft_label: mean={neg_labels.mean().item():.4f}, "
                              f"min={neg_labels.min().item():.4f}, "
                              f"max={neg_labels.max().item():.4f}")
            else:
                # 没有 distance 信息，回退到硬标签
                neg_labels = torch.zeros(batch_size, device=device)  # 硬负标签=0

            # 正样本二分类损失（logit + label=1）
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)  # 正样本 BCE
            # 负样本逐样本损失（logit + soft label）
            neg_loss_per_sample = F.binary_cross_entropy_with_logits(
                neg_logits, neg_labels, reduction='none'
            )  # [B] 逐样本负样本 BCE

            # 按 validity mask 屏蔽 episode 过短的样本
            if neg_z2_valid is not None:  # 仅使用有效负样本
                valid_mask = neg_z2_valid.float().to(device)
                n_valid = valid_mask.sum().clamp(min=1.0)  # 防止除零
                neg_loss = (neg_loss_per_sample * valid_mask).sum() / n_valid  # masked mean
            else:
                neg_loss = neg_loss_per_sample.mean()  # 全量平均

            # 正负样本对等权平均得到 ITM 总损失
            loss = (pos_loss + neg_loss) / 2.0  # 正负平均

            # 诊断信息
            with torch.no_grad():
                pos_acc = (pos_logits > 0).float().mean().item()  # 正样本判真率
                neg_acc = (neg_logits < 0).float().mean().item()  # 负样本判假率
                n_valid_int = int(neg_z2_valid.sum().item()) if neg_z2_valid is not None else batch_size  # 有效数
                avg_neg_soft_label = neg_labels.mean().item() if neg_distance is not None else 0.0  # 软标签均值

            info = {
                "pos_score": pos_logits.mean().item(),
                "neg_score_1": neg_logits.mean().item(),
                "neg_score_2": neg_logits.mean().item(),  # 兼容旧日志格式
                "pos_acc": pos_acc,
                "neg_acc": neg_acc,
                "avg_neg_soft_label": avg_neg_soft_label,
                "neg_type": "gt_temporal",
                "neg_valid_ratio": n_valid_int / batch_size,
            }
        else:
            raise ValueError(
                "neg_z2 must be provided. Batch cross-pair fallback has been removed. "
                "Both L1 and L2 should use GT temporal offset negatives."
            )

        return loss, info


class ForwardModel(nn.Module):
    """嵌入空间前向动力学模型：预测执行 action 后的观察嵌入。

    predicted_z3 = z1 + MLP(concat(z1, z2))
    残差设计：MLP 只学习 delta，z1 作为 baseline。
    """
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        # 功能总结：预测 z3（残差式前向动力学），只学习 delta。
        delta = self.mlp(torch.cat([z1, z2], dim=-1))
        return z1 + delta


def _clamp_embedding_norm(z: Tensor, max_norm: float = 0.0) -> Tensor:
    """Clamp embedding L2 norm to prevent unbounded growth.

    max_norm <= 0 时直接返回原始 z（MEAN_STD 模式不需要 clamp）。
    max_norm > 0 时，norm 超过阈值的 embedding 等比缩放到 max_norm（QUANTILES 建议 32.0）。
    """
    # 功能总结：限制 embedding 的 L2 范数，避免数值膨胀。
    if max_norm <= 0:
        return z
    z_norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return z * torch.clamp(max_norm / z_norm, max=1.0)


def contrastive_loss_with_structure(
        z1: Tensor,
        z2: Tensor,
        z2_original: Tensor,
        itm_head: nn.Module = None,  # ITM Head，替代 dot product 跨模态对比
        z3: Tensor = None,  # obs_{t+k} 特征，三元因果链第三节点
        z2_neg: Tensor = None,  # L1 负样本：同 episode GT 时序错位编码的 z2
        z2_neg_valid: Tensor = None,  # [B] bool，L1 neg 是否有效
        z2_neg_distance: Tensor = None,  # [B] float，L1 neg 的 action L2 距离
        z3_neg: Tensor = None,  # L2 负样本：同 episode 远离 t+k 的 GT obs 编码的 z3
        z3_neg_valid: Tensor = None,  # [B] bool，L2 neg 是否有效
        z3_neg_distance: Tensor = None,  # [B] float，L2 neg 的 obs 嵌入 L2 距离
        temperature: float = 0.07,
        temperature_structure: float = 0.07,
        temperature_intra: float = 0.2,  # 同模态对比的温度
        beta_intra: float = 1.0,  # action 软权重的温度
        top_k: int = 16,
        lambda_intra: float = 1.0,  # 同模态 loss 权重（基础值）
        lambda_cross: float = 1.0,  # 跨模态 loss 权重（基础值）
        lambda_structure: float = 1.0,
        lambda_anticollapse: float = 1.0,
        gamma: float = 0.4,
        eps: float = 1e-4,
        log_interval: int = 10,
        warmup_steps: int = 0,  # gating 提供自然 warmup，不需要显式 warmup
        decay_steps: int = 1000,
) -> tuple[Tensor, dict]:
    """对比学习损失函数。

    损失组成：
    1. 跨模态损失：ITM 三元因果链 obs_t ↔ action_t ↔ obs_{t+k}
       - L1: obs_t ↔ action_t，使用 GT 时序错位负样本（z2_neg）
       - L2: action_t ↔ obs_{t+k}，使用 GT 远离 t+k 的负样本（z3_neg）
    2. 结构保持损失：保持 z2 的 top_k 邻居排序一致性
    3. 防坍缩损失：variance + covariance

    Returns:
        total_loss: 标量损失
        loss_info: 各子项数值 dict，用于 wandb 日志
    """
    # 功能总结：计算 CMP 的三部分损失（跨模态/结构/防坍缩），并返回日志信息。
    # 使用函数属性来跟踪调用次数（静态变量）
    if not hasattr(contrastive_loss_with_structure, 'step_count'):
        contrastive_loss_with_structure.step_count = 0  # 首次调用初始化计数器
    contrastive_loss_with_structure.step_count += 1  # 步数递增
    step = contrastive_loss_with_structure.step_count  # 当前步数

    # 权重调度：intra 逐渐衰减，cross 保持不变
    if warmup_steps > 0 and step < warmup_steps:
        cross_progress = step / warmup_steps  # warmup 进度
        lambda_intra_eff = lambda_intra  # 同模态权重（当前未使用）
        lambda_cross_eff = lambda_cross * cross_progress  # 跨模态权重线性升温
    else:
        decay_progress = (step - warmup_steps) / decay_steps  # 衰减进度
        lambda_intra_eff = lambda_intra * max(0.3, 1.0 - decay_progress)  # 同模态权重衰减
        lambda_cross_eff = lambda_cross  # 跨模态权重保持

    # Flatten inputs
    z1_flat = z1 if z1.dim() == 2 else z1.reshape(-1, z1.shape[-1])  # [B, D]
    z2_flat = z2 if z2.dim() == 2 else z2.reshape(-1, z2.shape[-1])  # [B, D]
    batch_size = z1_flat.shape[0]  # 扁平后的 batch
    dim = z1_flat.shape[-1]  # 特征维度

    # Action 距离矩阵
    actions_flat = z2_original.reshape(batch_size, -1) if z2_original.dim() == 3 else z2_original  # 展平动作
    action_distances = torch.cdist(actions_flat, actions_flat, p=2)  # 动作 L2 距离矩阵
    eye_mask = torch.eye(batch_size, dtype=torch.bool, device=z2.device)  # 对角线 mask

    # 计算 z1/z2 内部相似度（用于监控）
    z1_norm = F.normalize(z1_flat, p=2, dim=-1)  # z1 归一化
    z1_sim = torch.matmul(z1_norm, z1_norm.T)  # 余弦相似度矩阵
    z1_internal_sim = z1_sim[~eye_mask].mean().item()  # 去除对角线平均

    z2_norm = F.normalize(z2_flat, p=2, dim=-1)  # z2 归一化
    z2_sim = torch.matmul(z2_norm, z2_norm.T)  # 余弦相似度矩阵
    z2_internal_sim = z2_sim[~eye_mask].mean().item()  # 去除对角线平均

    # ========== 1. 跨模态损失 (z1 vs z2) — ITM 三元因果链 ==========
    if itm_head is not None:
        # Flatten z3 for L2
        z3_flat = None  # L2 正样本（obs_{t+k}）
        if z3 is not None:
            z3_flat = z3 if z3.dim() == 2 else z3.reshape(-1, z3.shape[-1])

        # Flatten z2_neg for L1 GT 时序错位负样本
        z2_neg_flat = None  # L1 负样本（action 时序错位）
        if z2_neg is not None:
            z2_neg_flat = z2_neg if z2_neg.dim() == 2 else z2_neg.reshape(-1, z2_neg.shape[-1])

        # Flatten z3_neg for L2 GT 远离 t+k 的负样本
        z3_neg_flat = None  # L2 负样本（远离的未来观测）
        if z3_neg is not None:
            z3_neg_flat = z3_neg if z3_neg.dim() == 2 else z3_neg.reshape(-1, z3_neg.shape[-1])

        # L1: obs_t ↔ action_t — GT 时序错位负样本（同 episode 跨 chunk）
        cross_loss_L1, itm_info = itm_head.compute_loss(
            z1_flat, z2_flat, action_distances=action_distances,
            neg_z2=z2_neg_flat, neg_z2_valid=z2_neg_valid,
            neg_distance=z2_neg_distance,  # L1 action 距离
        )

        # L2: action_t ↔ obs_{t+k} — GT 远离 t+k 的负样本
        cross_loss_L2 = torch.tensor(0.0, device=z2.device)  # 默认无 L2
        itm_info_2 = None  # L2 诊断信息
        if z3_flat is not None:
            cross_loss_L2, itm_info_2 = itm_head.compute_loss(
                z2_flat, z3_flat,
                neg_z2=z3_neg_flat, neg_z2_valid=z3_neg_valid,
                neg_distance=z3_neg_distance,  # L2 obs 嵌入距离
            )

        # 汇总跨模态损失（L1+L2）
        cross_loss = cross_loss_L1 + cross_loss_L2

        # 诊断用：从 ITM info 提取 gap 信息
        with torch.no_grad():
            pos_mean = itm_info['pos_score']  # 正样本分数均值
            neg_mean = (itm_info['neg_score_1'] + itm_info['neg_score_2']) / 2  # 负样本分数均值
            cross_gap = pos_mean - neg_mean  # 正负差距

    # ========== 2. 结构保持损失 ==========
    orig_distances = action_distances.clone()  # 原始动作距离
    orig_distances[eye_mask] = float('inf')  # 屏蔽自距离
    max_k = min(top_k, batch_size - 1)  # 实际 k
    total_pairs = 0  # 统计总对数
    violated_pairs = 0  # 统计不一致对数

    if max_k <= 0:
        structure_loss = torch.tensor(0.0, device=z2.device, dtype=z2.dtype)
    else:
        _, orig_knn_indices = torch.topk(orig_distances, k=max_k, dim=-1, largest=False)  # 原始 KNN
        learned_distances = torch.cdist(z2_flat, z2_flat, p=2)  # 学到的距离
        structure_losses = []  # 每样本结构损失
        for i in range(batch_size):
            knn_indices = orig_knn_indices[i]  # 第 i 个样本的邻居
            orig_knn_dists = orig_distances[i, knn_indices]  # 原始邻居距离
            learned_knn_dists = learned_distances[i, knn_indices]  # 学到的邻居距离

            # Soft Rank
            orig_dists_expanded = orig_knn_dists.unsqueeze(0)  # [1, k]
            orig_dists_expanded_T = orig_knn_dists.unsqueeze(1)  # [k, 1]
            orig_diff = orig_dists_expanded_T - orig_dists_expanded  # 成对差值
            orig_soft_ranks = torch.sigmoid(orig_diff / temperature_structure).sum(dim=1)  # 软排序

            learned_dists_expanded = learned_knn_dists.unsqueeze(0)  # [1, k]
            learned_dists_expanded_T = learned_knn_dists.unsqueeze(1)  # [k, 1]
            learned_diff = learned_dists_expanded_T - learned_dists_expanded  # 成对差值
            learned_soft_ranks = torch.sigmoid(learned_diff / temperature_structure).sum(dim=1)  # 软排序

            rank_loss = F.l1_loss(learned_soft_ranks / (max_k - 1), orig_soft_ranks / (max_k - 1))  # 排名损失
            pair_loss = torch.mean((learned_diff.sign() - orig_diff.sign()).abs())  # 相对顺序一致性
            combined_loss = rank_loss + 0.1 * pair_loss  # 组合结构损失
            structure_losses.append(combined_loss)  # 收集

            # 统计排序一致性
            orig_ranks = torch.argsort(orig_knn_dists)  # 原始排序
            learned_ranks = torch.argsort(learned_knn_dists)  # 学到排序
            rank_matches = (orig_ranks == learned_ranks).sum().item()  # 排序一致数
            total_pairs += max_k  # 累计总对数
            violated_pairs += (max_k - rank_matches)  # 统计不一致对

        # 聚合结构损失（batch 平均）
        structure_loss = torch.mean(torch.stack(structure_losses))
        violation_rate = violated_pairs / total_pairs if total_pairs > 0 else 0.0  # 一致性比例

    # ========== 4. 防坍缩损失 ==========
    std_z1 = torch.sqrt(torch.var(z1_flat, dim=0) + eps)  # z1 标准差
    std_z2 = torch.sqrt(torch.var(z2_flat, dim=0) + eps)  # z2 标准差
    # 方差项：鼓励各维度方差不小于 gamma
    variance_loss = torch.mean(F.relu(gamma - std_z1)) + torch.mean(F.relu(gamma - std_z2))

    # 协方差项：惩罚特征维度间相关性（去掉对角线）
    z1_centered = z1_flat - torch.mean(z1_flat, dim=0, keepdim=True)  # 去均值
    z2_centered = z2_flat - torch.mean(z2_flat, dim=0, keepdim=True)  # 去均值
    cov_z1 = (z1_centered.T @ z1_centered) / (batch_size - 1)  # 协方差
    cov_z2 = (z2_centered.T @ z2_centered) / (batch_size - 1)  # 协方差

    off_diagonal_mask = 1 - torch.eye(dim, device=z1.device, dtype=z1.dtype)  # 非对角 mask
    offdiag_z1 = cov_z1 * off_diagonal_mask  # 去对角线
    offdiag_z2 = cov_z2 * off_diagonal_mask  # 去对角线
    cov_loss_z1 = torch.sum(torch.square(offdiag_z1)) / (dim * dim)  # z1 协方差损失
    cov_loss_z2 = torch.sum(torch.square(offdiag_z2)) / (dim * dim)  # z2 协方差损失
    covariance_loss = cov_loss_z1 + cov_loss_z2  # 协方差总损失

    # 防坍缩损失 = 方差项 + 协方差项
    anticollapse_loss = variance_loss + covariance_loss

    # ========== 总损失（使用动态权重）==========
    # intra_loss 已移除（不再使用同模态 RS-CL），仅保留 cross/structure/anticollapse
    weighted_cross_loss = lambda_cross_eff * cross_loss  # 跨模态加权
    weighted_structure_loss = lambda_structure * structure_loss  # 结构加权
    weighted_anticollapse_loss = lambda_anticollapse * anticollapse_loss  # 防坍缩加权

    # 三项损失线性加权得到 CMP 总损失
    total_loss = weighted_cross_loss + weighted_structure_loss + weighted_anticollapse_loss

    # ========== 日志输出 ==========
    step = contrastive_loss_with_structure.step_count  # 当前步数
    cross_mode = "ITM" if itm_head is not None else "dot"  # 跨模态类型
    logger = logging.getLogger(__name__)  # 日志器

    # 提取 soft label 信息
    soft_label_val = itm_info.get('avg_neg_soft_label', 0.0) if itm_info is not None else 0.0  # 软负标签均值

    # 每 10 步：简洁一行日志
    if step % log_interval == 0:
        total_val = total_loss.item()  # 当前总损失
        if itm_head is not None:
            logger.info(f"[CMP|{step:4d}] "
                  f"z1:{z1_internal_sim:.3f} z2:{z2_internal_sim:.3f} | "
                  f"gap:{cross_gap:+.3f}(ITM) soft_neg:{soft_label_val:.3f} | "
                  f"L:{total_val:.4f}")
        else:
            logger.info(f"[CMP|{step:4d}] "
                  f"z1:{z1_internal_sim:.3f} z2:{z2_internal_sim:.3f} | "
                  f"gap:{cross_gap:+.3f}(dot) | "
                  f"L:{total_val:.4f}")

    # 每 100 步：详细损失分解
    if step % 100 == 0:
        cross_val = cross_loss.item()  # 跨模态值
        structure_val = structure_loss.item()  # 结构值
        anticollapse_val = anticollapse_loss.item()  # 防坍缩值
        total_val = total_loss.item()  # 总损失

        detail_log = (
            f"\n{'='*70}\n"
            f"[Step {step}] CMP Loss Breakdown (cross: {cross_mode})\n"
            f"{'='*70}\n"
            f"  Cross-Modal ({cross_mode}): {cross_val:.4f} × {lambda_cross_eff:.2f} = {cross_val * lambda_cross_eff:.4f}\n"
        )
        if itm_info is not None:
            detail_log += (
                f"    L1 (obs_t↔act_t): pos_score={itm_info['pos_score']:.4f}, "
                f"neg_score={itm_info['neg_score_1']:.4f}/{itm_info['neg_score_2']:.4f}, "
                f"pos_acc={itm_info['pos_acc']:.3f}, soft_neg_label={itm_info.get('avg_neg_soft_label', 0):.3f}\n"
            )
            if itm_info_2 is not None:
                detail_log += (
                    f"    L2 (act_t↔obs_t+k): pos_score={itm_info_2['pos_score']:.4f}, "
                    f"neg_score={itm_info_2['neg_score_1']:.4f}/{itm_info_2['neg_score_2']:.4f}, "
                    f"pos_acc={itm_info_2['pos_acc']:.3f}\n"
                )
            if itm_info.get('neg_type') == 'gt_temporal':
                detail_log += (
                    f"    L1 Neg: GT temporal offset, "
                    f"valid_ratio={itm_info.get('neg_valid_ratio', 0):.2f}\n"
                )
        else:
            temp_val = temperature.item() if hasattr(temperature, 'item') else temperature
            detail_log += f"    pos: {pos_mean:.4f}, neg: {neg_mean:.4f}, gap: {cross_gap:+.4f}, T: {temp_val:.4f}\n"
        detail_log += f"  Structure (top-{top_k}):   {structure_val:.4f} × {lambda_structure:.2f} = {structure_val * lambda_structure:.4f}\n"
        if max_k > 0 and total_pairs > 0:
            consistent_rate = ((total_pairs - violated_pairs) / total_pairs) * 100  # 排序一致率
            detail_log += f"    ranking consistency: {consistent_rate:.1f}%\n"
        detail_log += (
            f"  Anti-collapse:        {anticollapse_val:.4f} × {lambda_anticollapse:.2f} = {anticollapse_val * lambda_anticollapse:.4f}\n"
            f"  Total Loss:           {total_val:.4f}\n"
            f"  Feature Stats:        z1_sim={z1_internal_sim:.4f} | z2_sim={z2_internal_sim:.4f}\n"
            f"{'='*70}"
        )
        logger.info(detail_log)

    # ========== 返回 loss + 子项 dict（供 wandb 日志） ==========
    loss_info = {
        "cmp/structure_loss": structure_loss.item(),
    }
    if itm_head is not None:
        loss_info["cmp/z1_z2_cross_loss"] = cross_loss_L1.item()
        loss_info["cmp/z2_z3_cross_loss"] = cross_loss_L2.item()
        loss_info["cmp/total_cross_loss"] = cross_loss.item()
        if itm_info.get('neg_type') == 'gt_temporal':
            loss_info["cmp/L1_neg_valid_ratio"] = itm_info.get('neg_valid_ratio', 0)
    else:
        loss_info["cmp/total_cross_loss"] = cross_loss.item()

    return total_loss, loss_info


def compute_vicreg_similarity(
        z1: Tensor,
        z2: Tensor,
        eps: float = 1e-4,
) -> Tensor:
    """Compute similarity based on VICReg invariance loss (L2 distance).

    Args:
        z1: First representation [batch, num_tokens, dim] or [batch, dim]
        z2: Second representation [batch, num_tokens, dim] or [batch, dim]
        eps: Small constant for numerical stability

    Returns:
        Similarity value (scalar tensor), lower means more similar
        This is the mean squared L2 distance between representations
    """
    # Handle both 2D and 3D inputs
    if z1.dim() == 2:
        z1 = z1.unsqueeze(1)  # [batch, dim] -> [batch, 1, dim]
    if z2.dim() == 2:
        z2 = z2.unsqueeze(1)  # [batch, dim] -> [batch, 1, dim]

    batch_size, num_tokens, dim = z1.shape

    # Reshape to (batch*num_tokens, dim)
    z1_flat = z1.reshape(batch_size, -1)  # [batch*num_tokens, dim]
    z2_flat = z2.reshape(batch_size, -1)  # [batch*num_tokens, dim]

    # Compute mean squared L2 distance (invariance loss component)
    similarity = torch.mean(torch.square(z1_flat - z2_flat), dim=-1)  # [batch]

    return similarity


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
            self,
            vlm_config,
            action_expert_config,
            use_adarms=None,
            precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
            self,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: list[torch.FloatTensor] | None = None,
            inputs_embeds: list[torch.FloatTensor] | None = None,
            use_cache: bool | None = None,
            adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                                                 hasattr(self.gemma_expert.model, "gradient_checkpointing")
                                                 and self.gemma_expert.model.gradient_checkpointing
                                                 and self.training
                                         ) or (hasattr(self,
                                                       "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    norm = models[i].norm
                    # Check if this is an adaRMS version (has cond_dim but no weight)
                    cond_dim = getattr(norm, 'cond_dim', None)
                    has_weight = hasattr(norm, 'weight')
                    is_adarms = cond_dim is not None and not has_weight

                    if is_adarms and adarms_cond[i] is None:
                        # For adaRMS version with None cond, provide a zero cond tensor
                        batch_size = hidden_states.shape[0]
                        zero_cond = torch.zeros(batch_size, cond_dim, device=hidden_states.device,
                                                dtype=hidden_states.dtype)
                        out_emb, _ = models[i].norm(hidden_states, cond=zero_cond)
                    elif adarms_cond[i] is not None:
                        out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    else:
                        out_emb, _ = models[i].norm(hidden_states)
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values

    def forward_partial(
            self,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: list[torch.FloatTensor] | None = None,
            inputs_embeds: list[torch.FloatTensor] | None = None,
            use_cache: bool | None = None,
            adarms_cond: list[torch.Tensor] | None = None,
            part_layer_num: int | None = None,
    ):
        """只处理前 part_layer_num 层，返回中间输出和状态。

        Args:
            attention_mask: 注意力掩码
            position_ids: 位置 ID
            past_key_values: 过去的 key-value cache
            inputs_embeds: 输入嵌入列表 [prefix_embeds, suffix_embeds]
            use_cache: 是否使用 cache
            adarms_cond: adaRMS 条件
            part_layer_num: 要处理的层数（从前开始）

        Returns:
            tuple: ([prefix_output, suffix_output], intermediate_state)
                   intermediate_state 包含继续处理所需的信息
        """
        if adarms_cond is None:
            adarms_cond = [None, None]

        if inputs_embeds[0] is None:
            raise ValueError(
                "forward_partial requires both prefix and suffix inputs. "
                "Use regular forward() for single-path processing."
            )

        if part_layer_num is None:
            raise ValueError("part_layer_num must be specified for forward_partial")

        models = [self.paligemma.language_model, self.gemma_expert.model]
        num_layers = self.paligemma.config.text_config.num_hidden_layers

        if part_layer_num > num_layers:
            raise ValueError(f"part_layer_num ({part_layer_num}) cannot exceed total layers ({num_layers})")

        # Check if gradient checkpointing is enabled
        use_gradient_checkpointing = (
                                             hasattr(self.gemma_expert.model, "gradient_checkpointing")
                                             and self.gemma_expert.model.gradient_checkpointing
                                             and self.training
                                     ) or (hasattr(self,
                                                   "gradient_checkpointing") and self.gradient_checkpointing and self.training)

        # Adjust attention_mask and position_ids if inputs_embeds contains None
        adjusted_attention_mask, adjusted_position_ids = _adjust_attention_and_position_for_none_inputs(
            inputs_embeds, attention_mask, position_ids
        )

        # Process only the first part_layer_num layers
        current_inputs_embeds = inputs_embeds
        for layer_idx in range(part_layer_num):
            if use_gradient_checkpointing:
                current_inputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete,
                    layer_idx,
                    current_inputs_embeds,
                    adjusted_attention_mask,
                    adjusted_position_ids,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    paligemma=self.paligemma,
                    gemma_expert=self.gemma_expert,
                )
            else:
                current_inputs_embeds = compute_layer_complete(
                    layer_idx,
                    current_inputs_embeds,
                    adjusted_attention_mask,
                    adjusted_position_ids,
                    adarms_cond,
                    paligemma=self.paligemma,
                    gemma_expert=self.gemma_expert,
                )

        # Return intermediate outputs and state for continuation
        intermediate_state = {
            "inputs_embeds": current_inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "adarms_cond": adarms_cond,
            "processed_layers": part_layer_num,
            "total_layers": num_layers,
        }

        return current_inputs_embeds, intermediate_state

    # def forward_remaining(
    #     self,
    #     intermediate_state: dict,
    #     use_cache: bool | None = None,
    # ):
    #     """从中间状态继续处理剩余的层。
    #
    #     Args:
    #         intermediate_state: 由 forward_partial 返回的中间状态字典
    #         use_cache: 是否使用 cache（保留用于接口一致性）
    #
    #     Returns:
    #         tuple: ([prefix_output, suffix_output], None)
    #     """
    #     inputs_embeds = intermediate_state["inputs_embeds"]
    #     attention_mask = intermediate_state["attention_mask"]
    #     position_ids = intermediate_state["position_ids"]
    #     adarms_cond = intermediate_state["adarms_cond"]
    #     processed_layers = intermediate_state["processed_layers"]
    #     total_layers = intermediate_state["total_layers"]
    #
    #     remaining_layers = total_layers - processed_layers
    #     if remaining_layers <= 0:
    #         raise ValueError(
    #             f"No remaining layers to process. Already processed {processed_layers} out of {total_layers}"
    #         )
    #
    #     models = [self.paligemma.language_model, self.gemma_expert.model]
    #
    #     # Check if gradient checkpointing is enabled
    #     use_gradient_checkpointing = (
    #         hasattr(self.gemma_expert.model, "gradient_checkpointing")
    #         and self.gemma_expert.model.gradient_checkpointing
    #         and self.training
    #     ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)
    #
    #     # Process remaining layers
    #     for layer_idx in range(processed_layers, total_layers):
    #         if use_gradient_checkpointing:
    #             inputs_embeds = torch.utils.checkpoint.checkpoint(
    #                 compute_layer_complete,
    #                 layer_idx,
    #                 inputs_embeds,
    #                 attention_mask,
    #                 position_ids,
    #                 adarms_cond,
    #                 use_reentrant=False,
    #                 preserve_rng_state=False,
    #                 paligemma=self.paligemma,
    #                 gemma_expert=self.gemma_expert,
    #             )
    #         else:
    #             inputs_embeds = compute_layer_complete(
    #                 layer_idx,
    #                 inputs_embeds,
    #                 attention_mask,
    #                 position_ids,
    #                 adarms_cond,
    #                 paligemma=self.paligemma,
    #                 gemma_expert=self.gemma_expert,
    #             )
    #
    #     # Apply final norm
    #     def compute_final_norms(inputs_embeds, adarms_cond):
    #         outputs_embeds = []
    #         for i, hidden_states in enumerate(inputs_embeds):
    #             norm = models[i].norm
    #             # Check if this is an adaRMS version (has cond_dim but no weight)
    #             cond_dim = getattr(norm, 'cond_dim', None)
    #             has_weight = hasattr(norm, 'weight')
    #             is_adarms = cond_dim is not None and not has_weight
    #
    #             if is_adarms and adarms_cond[i] is None:
    #                 # For adaRMS version with None cond, provide a zero cond tensor
    #                 batch_size = hidden_states.shape[0]
    #                 zero_cond = torch.zeros(batch_size, cond_dim, device=hidden_states.device, dtype=hidden_states.dtype)
    #                 out_emb, _ = models[i].norm(hidden_states, cond=zero_cond)
    #             elif adarms_cond[i] is not None:
    #                 out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
    #             else:
    #                 out_emb, _ = models[i].norm(hidden_states)
    #             outputs_embeds.append(out_emb)
    #         return outputs_embeds
    #
    #     # Apply gradient checkpointing to final norm if enabled
    #     if use_gradient_checkpointing:
    #         outputs_embeds = torch.utils.checkpoint.checkpoint(
    #             compute_final_norms,
    #             inputs_embeds,
    #             adarms_cond,
    #             use_reentrant=False,
    #             preserve_rng_state=False,
    #         )
    #     else:
    #         outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)
    #
    #     prefix_output = outputs_embeds[0]
    #     suffix_output = outputs_embeds[1]
    #
    #     return [prefix_output, suffix_output], None


class SingleHeadContentAttention(nn.Module):
    """单头内容注意力网络。

    输入: suffix_outs [batch_size, attn_act_len, input_dim] + 可学习的分类头
    输出: 分类头对应的输出 [batch_size, hidden_dim]

    实现了一个类似 Vision Transformer (ViT) 中 class token 的注意力机制，
    用于从动作序列中提取聚合特征。
    """

    def __init__(self, hidden_dim: int, input_dim: int, attn_act_len: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.attn_act_len = attn_act_len

        # 可学习的分类头（query），使用较小的初始化
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        # 单头注意力层（简单设计，层数不多）
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出层归一化和投影
        # 注意：LayerNorm 将在 forward 中应用在拼接后的序列上（包含 cls token）
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, suffix_outs: Tensor) -> Tensor:
        """前向传播。

        Args:
            suffix_outs: Tensor with shape [batch_size, attn_act_len, input_dim]

        Returns:
            output: [batch_size, hidden_dim] - 分类头对应的输出
        """
        batch_size = suffix_outs.shape[0]

        # 验证输入维度
        if suffix_outs.shape[1] != self.attn_act_len:
            raise ValueError(
                f"Expected suffix_outs to have {self.attn_act_len} tokens in dim 1, "
                f"but got {suffix_outs.shape[1]}"
            )
        if suffix_outs.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected suffix_outs to have input_dim={self.input_dim} in dim 2, "
                f"but got {suffix_outs.shape[2]}"
            )

        # 投影输入到 hidden_dim
        x = self.in_proj(suffix_outs)  # [batch_size, attn_act_len, hidden_dim]

        # 扩展 class token 到 batch size
        cls = self.class_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]

        # Pre-LN: 在注意力之前应用 LayerNorm
        # 拼接 cls token 和 content tokens，然后对整个序列做 LayerNorm
        x_normed = self.layer_norm(x)
        x_normed = torch.cat([cls, x_normed], dim=1)
        # 分离 cls token 和 content tokens
        cls_normed = x_normed[:, :1]  # [batch_size, 1, hidden_dim]
        content_normed = x_normed[:, 1:]  # [batch_size, attn_act_len, hidden_dim]

        # 计算 Q, K, V
        q = self.q_proj(cls_normed)  # [batch_size, 1, hidden_dim]
        k = self.k_proj(content_normed)  # [batch_size, attn_act_len, hidden_dim]
        v = self.v_proj(content_normed)  # [batch_size, attn_act_len, hidden_dim]

        # 单头注意力计算：缩放点积注意力
        scale = math.sqrt(self.hidden_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [batch_size, 1, attn_act_len]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, 1, attn_act_len]

        # 应用注意力权重
        attended = torch.matmul(attn_weights, v)  # [batch_size, 1, hidden_dim]
        attended = attended.squeeze(1)  # [batch_size, hidden_dim]

        # 输出投影
        output = self.out_proj(attended)  # [batch_size, hidden_dim]

        return output


class CrossModalAttention(nn.Module):
    """跨模态注意力模块，让一个模态"看到"另一个模态。

    核心思想：z1 和 z2 来自不同模态，直接比较难以建立配对关系。
    通过 cross-attention，让 z1 基于 batch 内所有 z2 生成 attended 表示，
    正样本对 (z1[i], z2[i]) 的 attention 应该更强。
    """

    def __init__(self, hidden_dim: int = 1024, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Q 来自一个模态，K/V 来自另一个模态
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm_q = nn.LayerNorm(hidden_dim)
        self.layer_norm_kv = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        """
        Args:
            query: [batch, hidden_dim] - 查询模态（如观察 z1）
            key_value: [batch, hidden_dim] - 键值模态（如动作 z2）
        Returns:
            attended: [batch, hidden_dim] - query 基于 key_value 的表示
        """
        batch_size = query.size(0)

        # LayerNorm
        q = self.layer_norm_q(query)  # [batch, hidden_dim]
        kv = self.layer_norm_kv(key_value)  # [batch, hidden_dim]

        # 投影
        Q = self.q_proj(q)  # [batch, hidden_dim]
        K = self.k_proj(kv)  # [batch, hidden_dim]
        V = self.v_proj(kv)  # [batch, hidden_dim]

        # 重塑为多头格式 [batch, num_heads, head_dim]
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)

        # 注意力计算：每个 query i 与 batch 内所有 key j 计算
        # Q: [batch_q, num_heads, head_dim]
        # K: [batch_k, num_heads, head_dim]
        # attn[i, h, j] = Q[i, h, :] @ K[j, h, :] / sqrt(head_dim)
        # 使用 einsum: 'bnh,cnh->bnc' 表示 batch_q x num_heads x batch_k
        attn_weights = torch.einsum('bnh,cnh->bnc', Q, K) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)  # 对 batch_k 维度 softmax
        attn_weights = self.dropout(attn_weights)

        # 加权求和：output[i, h, :] = sum_j attn[i, h, j] * V[j, h, :]
        # attn: [batch_q, num_heads, batch_k]
        # V: [batch_k, num_heads, head_dim]
        # 使用 einsum: 'bnc,cnh->bnh'
        attended = torch.einsum('bnc,cnh->bnh', attn_weights, V)

        # 重塑回 [batch, hidden_dim]
        attended = attended.reshape(batch_size, self.hidden_dim)
        attended = self.out_proj(attended)

        # 残差连接
        return query + attended


class ObsQueryBridge(nn.Module):
    """使用可学习的 Query tokens 从 VLM 序列中提取观测特征，生成 z1 (obs embedding)。"""

    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 512,
        num_queries: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.queries = nn.Parameter(torch.randn(num_queries, output_dim) * 0.02)
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)
        self.o_proj = nn.Linear(output_dim, output_dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))  # 初始为 0，逐渐学习
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)
        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, vlm_features: Tensor) -> Tensor:
        """vlm_features: [batch, seq_len, input_dim] -> [batch, output_dim]

        VLA-Adapter 风格的 gating（参照 MLPResNetBlock_Pro）：
        - VLM attention (queries → VLM)：无 gating，始终有效，提供输入依赖
        - Query refinement (queries → queries)：有 gating，可学习的 refinement
        - gating=0 时：z1 仍依赖 VLM features，可区分样本
        - gating 增大时：query 间交互增强，学习更复杂的模式
        """
        batch_size = vlm_features.size(0)
        seq_len = vlm_features.size(1)

        vlm_proj = self.input_proj(vlm_features)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        ratio_g = torch.tanh(self.gating_factor)

        # Q, K, V 投影
        Q = self.q_proj(queries)
        K_vlm = self.k_proj(vlm_proj)    # VLM attention K
        V_vlm = self.v_proj(vlm_proj)    # VLM attention V
        K_query = self.k_proj(queries)   # query refinement K
        V_query = self.v_proj(queries)   # query refinement V

        # 重塑为多头格式
        Q = Q.view(batch_size, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K_vlm = K_vlm.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V_vlm = V_vlm.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_query = K_query.view(batch_size, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        V_query = V_query.view(batch_size, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)

        # VLA-Adapter 风格：VLM attention 无 gating，query refinement 有 gating
        attn_scores_vlm = torch.matmul(Q, K_vlm.transpose(-2, -1))  # 无 gating，始终有效
        attn_scores_query = torch.matmul(Q, K_query.transpose(-2, -1)) * ratio_g  # 有 gating

        # 合并并 softmax
        attn_scores = torch.cat([attn_scores_vlm, attn_scores_query], dim=-1) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 合并 V 并计算输出
        V_combined = torch.cat([V_vlm, V_query], dim=2)
        attn_out = torch.matmul(attn_weights, V_combined)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, self.num_queries, self.output_dim)
        attn_out = self.o_proj(attn_out)

        # 残差 + FFN
        out = self.ffn(attn_out + queries)
        out = self.norm(out)
        out = out.mean(dim=1)
        out = self.output_proj(out)

        return out


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # 单头内容注意力网络（可选）
        attn_act_len = getattr(config, "attn_act_len", None)
        origin_action_dim = self.config.output_features[ACTION].shape[0]
        if attn_act_len is not None and attn_act_len > 0:
            self.content_attention = SingleHeadContentAttention(
                hidden_dim=action_expert_config.width,
                input_dim=origin_action_dim,
                attn_act_len=attn_act_len,
            )
        else:
            self.content_attention = None

        # 可学习的分类头参数（用于对比学习）
        part_layer_num = getattr(config, "part_layer_num", None)
        if part_layer_num is not None and part_layer_num > 0:
            self.cls_head_prefix = nn.Parameter(
                torch.randn(1, 1, paligemma_config.width)
            )
            self.part_layer_num = part_layer_num
        else:
            self.cls_head_prefix = None
            self.part_layer_num = None
        self.cmp_step = 0

        # Velocity distillation: frozen anchor Expert state (initialized before online training)
        self._anchor_expert_state: dict[str, torch.Tensor] | None = None

        # 投影层
        self.proj_z1 = nn.Sequential(
            nn.Linear(paligemma_config.width, paligemma_config.width),
            nn.GELU(),
            nn.Linear(paligemma_config.width, 512),
        )
        self.proj_z2 = nn.Sequential(
            nn.Linear(action_expert_config.width, action_expert_config.width),
            nn.GELU(),
            nn.Linear(action_expert_config.width, 512),
        )

        # ObsQueryBridge: 从 VLM 序列提取动作相关特征
        self.obs_query_bridge = ObsQueryBridge(
            input_dim=paligemma_config.width,
            output_dim=512,
            num_queries=64,
            num_heads=8,
            dropout=0.1,
        )

        self.cmp_projection = self.proj_z1

        # ITM Head (ALBEF style) — 替代 dot product 做跨模态匹配
        self.itm_head = ITMHead(hidden_dim=512, proj_dim=256, dropout=0.1)

        # ForwardModel: 嵌入空间前向动力学（服务 replan 检测）
        self.forward_model = ForwardModel(hidden_dim=512)

        self.infonce_log_inv_temp = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

        self.cross_modal_attention = CrossModalAttention(
            hidden_dim=512,
            num_heads=8,
            dropout=0.1,
        )

        # z1 → z2 predictor：将 z1 映射到 z2 空间
        self.z1_to_z2_predictor = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
        )

        self._cross_modal_attention_deprecated = CrossModalAttention(
            hidden_dim=512,
            num_heads=8,
            dropout=0.1,
        )

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

        msg = """An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""

        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
            self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep, condition_token=None):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing.

        Args:
            noisy_actions: [batch, chunk_size, action_dim] 噪声动作
            timestep: [batch] 时间步
            condition_token: [batch, hidden_dim] 可选的条件 token（来自 prev_actions 的特征）
                            如果提供，会作为第一个 token 添加到 suffix embeddings 中
        """
        embs = []
        pad_masks = []
        att_masks = []

        bsize = noisy_actions.shape[0]

        # 如果有 condition token，先添加它作为第一个 token
        if condition_token is not None:
            # condition_token: [batch, hidden_dim] -> [batch, 1, hidden_dim]
            condition_emb = condition_token.unsqueeze(1)
            embs.append(condition_emb)
            condition_mask = torch.ones(bsize, 1, dtype=torch.bool, device=timestep.device)
            pad_masks.append(condition_mask)
            # condition token 使用 att_mask=1，prefix 不会 attend 到它
            # 但后续的 action tokens 可以 attend 到它
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        action_time_dim = action_time_emb.shape[1]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None,
                condition_token=None, return_internals=False) -> Tensor | tuple[Tensor, dict]:
        """Do a full training forward pass and compute the loss.

        Args:
            images: 图像列表
            img_masks: 图像掩码列表
            tokens: 语言 tokens
            masks: 语言 token 掩码
            actions: ground truth 动作 [batch, chunk_size, action_dim]
            noise: 可选的噪声
            time: 可选的时间步
            condition_token: [batch, hidden_dim] 可选的条件 token（来自 prev_actions 的特征）
            return_internals: 如果 True，额外返回 v_t/x_t/time/prefix_embs（用于 anchor loss）
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time, condition_token)

        if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        loss = F.mse_loss(u_t, v_t, reduction="none")
        if return_internals:
            return loss, {
                "v_t": v_t,
                "x_t": x_t,
                "time": time,
                "prefix_embs": prefix_embs,
                "prefix_pad_masks": prefix_pad_masks,
                "prefix_att_masks": prefix_att_masks,
            }
        return loss

    def _forward_anchor(self, internals: dict, condition_token=None) -> torch.Tensor:
        """Compute velocity prediction using frozen anchor Expert params.

        Uses param.data pointer swap (zero-copy), reuses VLM prefix_embs.
        The autograd graph of the caller's v_t is unaffected by the swap.
        """
        if self._anchor_expert_state is None:
            raise RuntimeError("Anchor Expert not initialized. Call init_anchor_expert() first.")

        x_t = internals["x_t"]
        time = internals["time"]
        prefix_embs = internals["prefix_embs"].detach()
        prefix_pad_masks = internals["prefix_pad_masks"]
        prefix_att_masks = internals["prefix_att_masks"]

        # 1. Save current param.data pointers (zero-copy, just pointer swap)
        saved = {}
        expert_modules = [
            ("gemma_expert.", self.paligemma_with_expert.gemma_expert),
            ("action_in_proj.", self.action_in_proj),
            ("action_out_proj.", self.action_out_proj),
            ("time_mlp_in.", self.time_mlp_in),
            ("time_mlp_out.", self.time_mlp_out),
        ]
        for prefix, module in expert_modules:
            for name, param in module.named_parameters():
                key = prefix + name
                saved[key] = param.data
                param.data = self._anchor_expert_state[key]

        # 2. Forward with frozen anchor Expert (no grad, reuse VLM prefix)
        with torch.no_grad():
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
                self.embed_suffix(x_t, time, condition_token)
            )
            if prefix_embs.dtype == torch.bfloat16:
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = suffix_out[:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_anchor = self.action_out_proj(suffix_out)

        # 3. Restore current param.data pointers
        for prefix, module in expert_modules:
            for name, param in module.named_parameters():
                key = prefix + name
                param.data = saved[key]

        return v_anchor

    def forward_cmp(
            self,
            images,
            img_masks,
            tokens,
            masks,
            future_images=None,
            future_img_masks=None,
            neg_gt_actions=None,
            neg_action_valid=None,
            neg_future_images=None,
            neg_future_img_masks=None,
            neg_future_valid=None,
            neg_action_distance=None,  # L1 负样本 action 距离
    ) -> Tensor:
        """三元因果链对比学习: obs_t ↔ action_t ↔ obs_{t+k}

        使用 ALBEF ITM (Image-Text Matching) 做匹配判断，不依赖 dot product。

        Args:
            images: 当前帧图像列表 (obs_t)
            img_masks: 当前帧图像掩码列表
            tokens: 语言 tokens
            masks: 语言 token 掩码
            future_images: 未来帧图像列表 (obs_{t+k})，None 时退化为二元
            future_img_masks: 未来帧图像掩码列表
            neg_gt_actions: [B, att_len, action_dim] 同 episode GT 时序错位动作（L1 负样本）
            neg_action_valid: [B] bool，每个样本 neg 是否有效
            neg_future_images: 同 episode 远离 t+k 的 GT obs 图像列表（L2 负样本）
            neg_future_img_masks: neg_future 图像掩码列表
            neg_future_valid: [B] bool，每个样本 neg_future 是否有效

        Returns:
            对比学习损失 (expanded to [batch_size, 1, action_dim])
        """
        # 功能总结：构造 z1/z2/z3 及其负样本，计算 CMP 各子项损失，并附加前向动力学损失。
        if self.cls_head_prefix is None:
            raise ValueError(
                "cls_head_prefix is not initialized. Please set part_layer_num in config."
            )
        if self.content_attention is None:
            raise ValueError(
                "content_attention is not initialized. Please set attn_act_len in config."
            )
        if self.part_layer_num is None:
            raise ValueError(
                "part_layer_num is not set in config."
            )

        batch_size = tokens.shape[0]  # batch 大小
        device = tokens.device  # 设备

        # ========== VLM 前向（共用辅助函数） ==========
        action_expert_config = get_gemma_config(self.config.action_expert_variant)  # expert 配置
        cls_head = self.cls_head_prefix.expand(batch_size, -1, -1)  # 复制 CLS token
        cls_pad_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)  # CLS pad mask
        cls_att_mask = torch.tensor([0], dtype=torch.bool, device=device)  # CLS attention mask
        dummy_suffix_embs = torch.zeros(
            batch_size, 1, action_expert_config.width, dtype=cls_head.dtype, device=device
        )
        dummy_suffix_pad_masks = torch.ones(batch_size, 1, dtype=torch.bool, device=device)  # 虚拟 suffix pad
        dummy_suffix_att_masks = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)  # 虚拟 suffix att

        def forward_partial_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids):
            intermediate_embeds, intermediate_state = self.paligemma_with_expert.forward_partial(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                adarms_cond=[None, None],
                part_layer_num=self.part_layer_num,
            )
            return intermediate_embeds, intermediate_state

        def encode_obs(imgs, img_msks):
            """将一组图像通过 VLM + ObsQueryBridge 编码为 z [B, 512]"""
            # 1) 取前缀 embedding（图像+语言）
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                imgs, img_msks, tokens, masks
            )
            # 2) 拼接 cls token，形成用于 CMP 的序列输入
            prefix_embs_with_cls = torch.cat(
                [cls_head.to(dtype=prefix_embs.dtype), prefix_embs], dim=1
            )
            prefix_pad_masks_with_cls = torch.cat([cls_pad_mask, prefix_pad_masks], dim=1)
            prefix_att_masks_with_cls = torch.cat(
                [cls_att_mask.unsqueeze(0).expand(batch_size, -1), prefix_att_masks], dim=1
            )
            # 3) 构造 attention mask
            pad_msks = torch.cat([prefix_pad_masks_with_cls, dummy_suffix_pad_masks], dim=1)
            att_msks = torch.cat([prefix_att_masks_with_cls, dummy_suffix_att_masks], dim=1)
            att_2d = make_att_2d_masks(pad_msks, att_msks)
            pos_ids = torch.cumsum(pad_msks, dim=1) - 1
            att_4d = self._prepare_attention_masks_4d(att_2d)

            # 4) 仅前向到 part_layer_num，得到中间特征
            inter_embeds, _ = self._apply_checkpoint(
                forward_partial_func, prefix_embs_with_cls, dummy_suffix_embs, att_4d, pos_ids
            )
            # vlm_seq = inter_embeds[0]  # 原版：CMP 梯度回传到 VLM front 6 layers
            # 5) 关键：CMP 梯度在此处截断，VLM 仅由 BC 更新
            vlm_seq = inter_embeds[0].detach()  # 切断 CMP → VLM 梯度，BC 独占 VLM 更新
            # 6) 通过 ObsQueryBridge 得到 z 表示
            z = self.obs_query_bridge(vlm_seq).to(dtype=torch.float32)
            # 7) 可选的 embedding 范数裁剪
            return _clamp_embedding_norm(z, self.config.embedding_max_norm)

        # ========== z1: obs_t ==========
        cmp_vec_0 = encode_obs(images, img_masks)  # [B, 512]
        if cmp_vec_0.dim() != 2:
            raise ValueError(f"Expected cmp_vec_0 to be 2D, got shape {cmp_vec_0.shape}")

        # ========== z2: action_t → ContentAttention → proj_z2 ==========
        attn_act_len = self.content_attention.attn_act_len  # 取前几步动作
        actions = self.sample_actions(images, img_masks, tokens, masks)  # 采样动作序列
        origin_action_dim = self.config.output_features[ACTION].shape[0]  # 原始动作维度
        selected_actions = actions[:, :attn_act_len, :origin_action_dim]  # 选取用于 CMP 的动作
        cmp_vec_1_raw = self.content_attention(selected_actions)  # 动作聚合表征
        # action 表征投影到 CMP 共享空间，并进行可选范数裁剪
        cmp_vec_1 = _clamp_embedding_norm(self.proj_z2(cmp_vec_1_raw).to(dtype=torch.float32), self.config.embedding_max_norm)

        # ========== z2_neg: 同 episode GT 时序错位负样本 ==========
        cmp_vec_1_neg = None  # L1 负样本表征
        neg_valid = None  # L1 负样本有效标记
        if neg_gt_actions is not None and neg_action_valid is not None:
            # neg_gt_actions: [B, att_len, action_dim]
            cmp_vec_1_neg_raw = self.content_attention(neg_gt_actions)
            cmp_vec_1_neg = _clamp_embedding_norm(self.proj_z2(cmp_vec_1_neg_raw).to(dtype=torch.float32), self.config.embedding_max_norm)
            neg_valid = neg_action_valid  # [B] bool

        # ========== z3: obs_{t+k} ==========
        cmp_vec_2 = None  # z3：未来观测表征
        if future_images is not None and future_img_masks is not None:
            cmp_vec_2 = encode_obs(future_images, future_img_masks)

        # ========== z3_neg: 同 episode 远离 t+k 的 GT obs（L2 负样本）==========
        cmp_vec_2_neg = None  # z3_neg：远离未来观测
        neg_future_valid_flag = None  # L2 负样本有效标记
        if neg_future_images is not None and neg_future_img_masks is not None:
            cmp_vec_2_neg = encode_obs(neg_future_images, neg_future_img_masks)
            neg_future_valid_flag = neg_future_valid  # [B] bool

        # 确保维度匹配
        if cmp_vec_0.shape != cmp_vec_1.shape:
            raise ValueError(
                f"Dimension mismatch: cmp_vec_0 shape {cmp_vec_0.shape} != cmp_vec_1 shape {cmp_vec_1.shape}"
            )

        # ========== 计算 L2 负样本距离（obs 嵌入 L2 距离）==========
        neg_future_distance = None  # L2 负样本距离
        if cmp_vec_2 is not None and cmp_vec_2_neg is not None:
            # [B] 每个样本的 obs 嵌入 L2 距离
            neg_future_distance = torch.norm(cmp_vec_2 - cmp_vec_2_neg, p=2, dim=-1)

            # Debug logging for L2 embedding distance
            if hasattr(self, 'cmp_step') and self.cmp_step % 100 == 0:
                print(f"[forward_cmp L2] step={self.cmp_step}, "
                      f"neg_future_distance mean={neg_future_distance.mean().item():.4f}, "
                      f"min={neg_future_distance.min().item():.4f}, "
                      f"max={neg_future_distance.max().item():.4f}")
                if neg_action_distance is not None:
                    print(f"[forward_cmp L1] step={self.cmp_step}, "
                          f"neg_action_distance mean={neg_action_distance.mean().item():.4f}, "
                          f"min={neg_action_distance.min().item():.4f}, "
                          f"max={neg_action_distance.max().item():.4f}")

        # ========== 调用 contrastive_loss_with_structure ==========
        loss_scalar, cmp_loss_info = contrastive_loss_with_structure(  # CMP 主损失
            z1=cmp_vec_0,
            z2=cmp_vec_1,
            z2_original=selected_actions,
            itm_head=self.itm_head,
            z3=cmp_vec_2,
            z2_neg=cmp_vec_1_neg,
            z2_neg_valid=neg_valid,
            z2_neg_distance=neg_action_distance,  # L1 负样本 action 距离
            z3_neg=cmp_vec_2_neg,
            z3_neg_valid=neg_future_valid_flag,
            z3_neg_distance=neg_future_distance,  # L2 负样本 obs 嵌入距离
        )

        # ========== L_forward: 前向动力学预测 loss（cosine loss）==========
        # 用 cosine loss 而非 MSE：只关注方向预测，不受 embedding norm 膨胀影响
        if cmp_vec_2 is not None:  # z3 存在时才计算
            # 仅更新 forward_model：输入与目标均 detach
            predicted_z3 = self.forward_model(cmp_vec_0.detach(), cmp_vec_1.detach())
            cos_sim = F.cosine_similarity(predicted_z3, cmp_vec_2.detach(), dim=-1)
            l_forward = (1.0 - cos_sim).mean()
            # 将前向动力学损失叠加到 CMP 总损失
            loss_scalar = loss_scalar + l_forward

            with torch.no_grad():
                forward_l2 = torch.norm(predicted_z3 - cmp_vec_2.detach(), dim=-1).mean()
                z1_norm = torch.norm(cmp_vec_0.detach(), dim=-1).mean()
                z3_norm = torch.norm(cmp_vec_2.detach(), dim=-1).mean()
                pred_z3_norm = torch.norm(predicted_z3, dim=-1).mean()
            cmp_loss_info["cmp/l_forward"] = l_forward.item()
            cmp_loss_info["cmp/forward_cos_sim"] = cos_sim.mean().item()
            cmp_loss_info["cmp/forward_l2"] = forward_l2.item()
            cmp_loss_info["cmp/z1_norm"] = z1_norm.item()
            cmp_loss_info["cmp/z3_norm"] = z3_norm.item()
            cmp_loss_info["cmp/pred_z3_norm"] = pred_z3_norm.item()

            # 每 10 步打印诊断
            if self.cmp_step % 10 == 0:
                _fwd_logger = logging.getLogger(__name__)
                if self.cmp_step % 100 == 0:
                    _fwd_logger.info(
                        f"  L_forward (cosine):   {l_forward.item():.4f}\n"
                        f"  cos_sim(pred,z3):     {cos_sim.mean().item():.4f}\n"
                        f"  Forward L2 distance:  {forward_l2.item():.4f}\n"
                        f"  ‖z1‖={z1_norm.item():.2f}  ‖z3‖={z3_norm.item():.2f}  ‖pred_z3‖={pred_z3_norm.item():.2f}\n"
                        f"  Total+Forward:        {loss_scalar.item():.4f}"
                    )
                else:
                    _fwd_logger.info(
                        f"[ForwardModel|{self.cmp_step:4d}] "
                        f"l_fwd:{l_forward.item():.4f} cos:{cos_sim.mean().item():.4f} "
                        f"‖z1‖:{z1_norm.item():.2f} ‖z3‖:{z3_norm.item():.2f} ‖pred‖:{pred_z3_norm.item():.2f}"
                    )

        # 扩展为 [batch_size, 1, action_dim] 以匹配 forward 返回格式
        action_dim = self.config.max_action_dim
        losses = loss_scalar.view(1, 1, 1).expand(batch_size, 1, action_dim)  # 扩展形状

        # 递增 CMP 步数计数器（用于日志/调度）
        self.cmp_step += 1
        return losses, cmp_loss_info

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
            self,
            images,
            img_masks,
            tokens,
            masks,
            noise=None,
            num_steps=None,
            prefix_embs=None,
            prefix_pad_masks=None,
            prefix_att_masks=None,
            condition_token=None,
            **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action.

        Args:
            images, img_masks, tokens, masks: 输入数据
            noise: 可选的初始噪声
            num_steps: 采样步数
            prefix_embs, prefix_pad_masks, prefix_att_masks: 可选的预计算 embed_prefix 结果，
                如果传入则跳过内部的 embed_prefix 调用（用于优化重复计算）
            condition_token: [batch, hidden_dim] 可选的条件 token（来自 prev_actions 的 content_attention），
                与 sample_actions_differentiable 保持一致
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        # 如果传入了预计算的 prefix_embs，则复用；否则调用 embed_prefix
        if prefix_embs is None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            # Define a closure function to properly capture expanded_time
            # This avoids the lambda expression (E731) and loop variable binding (B023) issues
            def denoise_step_partial_call(input_x_t, current_timestep=expanded_time):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                    condition_token=condition_token,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            # Euler step
            x_t += dt * v_t

            # Record x_t and v_t after Euler step
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

            time += dt

        return x_t

    def denoise_step(
            self,
            prefix_pad_masks,
            past_key_values,
            x_t,
            timestep,
            condition_token=None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep.

        Args:
            prefix_pad_masks: prefix 的 padding mask
            past_key_values: 缓存的 key-value
            x_t: 当前噪声状态 [batch, chunk_size, action_dim]
            timestep: 当前时间步 [batch]
            condition_token: [batch, hidden_dim] 可选的条件 token
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep, condition_token)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def sample_actions_differentiable(
            self,
            images,
            img_masks,
            tokens,
            masks,
            condition_token=None,
            noise=None,
            num_steps=None,
    ) -> Tensor:
        """可微分的动作采样方法，用于 online 训练。

        与 sample_actions 类似，但不使用 @torch.no_grad()，允许梯度回传。

        Args:
            images: 图像列表
            img_masks: 图像掩码列表
            tokens: 语言 tokens
            masks: 语言 token 掩码
            condition_token: [batch, hidden_dim] 可选的条件 token
            noise: 可选的初始噪声
            num_steps: 采样步数

        Returns:
            生成的动作 [batch, chunk_size, action_dim]
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # 注意：这里不使用 past_key_values 缓存，因为需要保持梯度流
        # 每一步都重新计算 prefix，但这是必要的代价

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            # 完整的 forward pass（不使用 KV cache）
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                x_t, expanded_time, condition_token
            )

            if (
                    self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                    == torch.bfloat16
            ):
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                prefix_embs_bf16 = prefix_embs.to(dtype=torch.bfloat16)
            else:
                prefix_embs_bf16 = prefix_embs

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs_bf16, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = suffix_out[:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)

            # Euler step
            x_t = x_t + dt * v_t
            time = time + dt

        return x_t


class PI05Policy(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
            self,
            config: PI05Config,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

        self.offline_mode: bool = False

    @classmethod
    def from_pretrained(
            cls: builtins.type[T],
            pretrained_name_or_path: str | Path,
            *,
            config: PreTrainedConfig | None = None,
            force_download: bool = False,
            resume_download: bool | None = None,
            proxies: dict | None = None,
            token: str | bool | None = None,
            cache_dir: str | Path | None = None,
            local_files_only: bool = False,
            revision: str | None = None,
            strict: bool = True,
            **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Now manually load and remap the state dict
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    use_auth_token=kwargs.get("use_auth_token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                    if remap_count <= 10:  # Only print first 10 to avoid spam
                        print(f"Remapped: {key} -> {new_key}")
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Load the remapped state dict into the model
            try:
                missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)
            except Exception as load_err:
                # When strict=True and checkpoint lacks CMP modules (cls_head, itm_head, etc.), retry with strict=False
                if strict:
                    print(
                        f"Strict load failed ({load_err}). Retrying with strict=False to load matching keys "
                        "(missing CMP modules will remain randomly initialized)."
                    )
                    missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
                else:
                    raise

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")
            elif missing_keys:
                print("Loaded matching keys; remaining missing keys (e.g. CMP modules) left at random init.")

        except Exception as e:
            print(f"Warning: Could not remap state dict keys: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
            self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                    r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                    key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def get_online_optim_params(self):
        """返回 online 训练需要的参数。

        Returns:
            包含 action_expert 相关参数的迭代器
        """
        online_params = []

        # gemma_expert 的参数
        if hasattr(self.model, 'paligemma_with_expert'):
            online_params.extend(
                self.model.paligemma_with_expert.gemma_expert.parameters()
            )

        # action_in_proj 和 action_out_proj 的参数
        if hasattr(self.model, 'action_in_proj'):
            online_params.extend(self.model.action_in_proj.parameters())
        if hasattr(self.model, 'action_out_proj'):
            online_params.extend(self.model.action_out_proj.parameters())

        # time_mlp 的参数
        if hasattr(self.model, 'time_mlp_in'):
            online_params.extend(self.model.time_mlp_in.parameters())
        if hasattr(self.model, 'time_mlp_out'):
            online_params.extend(self.model.time_mlp_out.parameters())

        return online_params

    def get_offline_optim_params(self):
        """返回 offline 训练需要的参数（所有参数）。

        Returns:
            包含所有参数的迭代器
        """
        return self.parameters()

    # ==================== Velocity Distillation (Anchor) ====================

    def _iter_expert_params(self):
        """Yield (flat_key, nn.Parameter) for all Expert parameters."""
        modules = [
            ("gemma_expert.", self.model.paligemma_with_expert.gemma_expert),
            ("action_in_proj.", self.model.action_in_proj),
            ("action_out_proj.", self.model.action_out_proj),
            ("time_mlp_in.", self.model.time_mlp_in),
            ("time_mlp_out.", self.model.time_mlp_out),
        ]
        for prefix, module in modules:
            for name, param in module.named_parameters():
                yield prefix + name, param

    def init_anchor_expert(self):
        """Snapshot current Expert params as frozen anchor for velocity distillation.
        Call after model loading, before online training begins."""
        state = {}
        for key, param in self._iter_expert_params():
            state[key] = param.data.detach().clone()  # stays on GPU (~600MB for gemma_300m)
        self.model._anchor_expert_state = state
        n_params = sum(v.numel() for v in state.values())
        mem_mb = sum(v.nbytes for v in state.values()) / (1024 ** 2)
        logging.info(f"[Anchor] Snapshotted {n_params:,} Expert params ({mem_mb:.0f} MB)")

    def _apply_param_cmp(self) -> None:
        self.offline_mode = True
        params = []

        # content_attention 的参数
        if self.model.content_attention is not None:
            params.extend(self.model.content_attention.parameters())

        # 投影层
        if self.model.proj_z1 is not None:
            params.extend(self.model.proj_z1.parameters())

        if self.model.proj_z2 is not None:
            params.extend(self.model.proj_z2.parameters())

        # ObsQueryBridge
        if hasattr(self.model, 'obs_query_bridge') and self.model.obs_query_bridge is not None:
            params.extend(self.model.obs_query_bridge.parameters())

        # ITM Head (ALBEF style)
        if hasattr(self.model, 'itm_head') and self.model.itm_head is not None:
            params.extend(self.model.itm_head.parameters())

        # cls_head_prefix 参数
        if self.model.cls_head_prefix is not None:
            params.append(self.model.cls_head_prefix)

        # ForwardModel（嵌入空间前向动力学）
        if hasattr(self.model, 'forward_model') and self.model.forward_model is not None:
            params.extend(self.model.forward_model.parameters())

        addition_ids = {id(p) for p in params}
        for p in self.parameters():
            p.requires_grad = id(p) in addition_ids

    def _apply_offline_params(self) -> None:
        self.offline_mode = True
        for p in self.parameters():
            p.requires_grad = True

    def _apply_online_params(self) -> None:
        self.offline_mode = False
        # 为 online 训练设置参数：只训练 action_expert的参数。
        online_params = []

        # gemma_expert 的参数
        if hasattr(self.model, 'paligemma_with_expert'):
            online_params.extend(
                self.model.paligemma_with_expert.gemma_expert.parameters()
            )

        # action_in_proj 和 action_out_proj 的参数
        if hasattr(self.model, 'action_in_proj'):
            online_params.extend(self.model.action_in_proj.parameters())
        if hasattr(self.model, 'action_out_proj'):
            online_params.extend(self.model.action_out_proj.parameters())

        # time_mlp 的参数
        if hasattr(self.model, 'time_mlp_in'):
            online_params.extend(self.model.time_mlp_in.parameters())
        if hasattr(self.model, 'time_mlp_out'):
            online_params.extend(self.model.time_mlp_out.parameters())

        # if self.model.content_attention is not None:
        #     online_params.extend(self.model.content_attention.parameters())

        # 设置所有参数的 requires_grad
        online_param_ids = {id(p) for p in online_params}
        for p in self.parameters():
            p.requires_grad = id(p) in online_param_ids

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

        # ITM score 记录（用于离线分析）
        self._itm_scores: list[Tensor] = []           # fresh ITM（每次生成新 chunk 时）
        self._stale_itm_scores: list[Tensor] = []     # stale ITM（用旧 chunk 的后续段）
        self._stale_itm_chunk_ids: list[int] = []     # 每个 stale score 对应的 chunk index
        self._stale_itm_segment_ids: list[int] = []   # 每个 stale score 对应的 segment index (1-4)
        self._full_action_chunk: Tensor | None = None  # 保存完整 chunk_size 的 action chunk
        self._chunk_segment_index: int = 0             # 当前 chunk 内已使用到第几段 (0=fresh刚生成, 1-4=stale)
        self._prev_chunk_actions: Tensor | None = None  # 上一 chunk 末尾 attn_act_len 步动作，用于 condition_token
        self._current_chunk_idx: int = -1              # 当前 chunk 编号
        self._select_action_count = 0

        # ForwardModel drift 跟踪
        self._predicted_z3: Tensor | None = None          # 上一步 ForwardModel 的预测
        self._drift_scores: list[Tensor] = []              # drift cosine distance [batch_size] per segment
        self._drift_chunk_ids: list[int] = []             # 对应 chunk index
        self._drift_segment_ids: list[int] = []           # 对应 segment index
        self._replan_count: int = 0                        # drift 触发 replan 的次数
        self._force_replan_next: bool = False                 # 自适应步长：中等 drift 触发的延迟 replan 标志

        # 模态对齐分析：z1(obs) 和 z2(action) 嵌入
        self._z1_embeddings: list[Tensor] = []  # z1 观测嵌入，每次生成新 chunk 时 [B, 512]
        self._z2_embeddings: list[Tensor] = []  # z2 动作嵌入，每次生成新 chunk 时 [B, 512]

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_single_image(self, img: Tensor, device: torch.device) -> Tensor:
        """预处理单张图像: device, dtype, resize, normalize."""
        if img.device != device:
            img = img.to(device)
        if img.dtype != torch.float32:
            img = img.to(torch.float32)

        is_channels_first = img.shape[1] == 3
        if is_channels_first:
            img = img.permute(0, 2, 3, 1)

        if img.shape[1:3] != self.config.image_resolution:
            img = resize_with_pad_torch(img, *self.config.image_resolution)

        img = img * 2.0 - 1.0

        if is_channels_first:
            img = img.permute(0, 3, 1, 2)

        return img

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].

        当 observation_delta_indices 启用时，images 为 [B, T, C, H, W]。
        此方法只返回 t=0 的当前帧。未来帧通过 _preprocess_future_images 获取。
        """
        images = []
        img_masks = []

        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]

            # 处理多时间步: [B, T, C, H, W] → 取 t=0 → [B, C, H, W]
            if img.dim() == 5:
                img = img[:, 0]

            img = self._preprocess_single_image(img, device)
            images.append(img)

            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def _preprocess_future_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]] | None:
        """提取 t+k 未来帧图像（三元因果链用）。

        当 observation_delta_indices = [0, k] 时，images 为 [B, 2, C, H, W]，
        此方法返回 t=1（即 obs_{t+k}）的图像。
        如果没有多时间步，返回 None。
        """
        device = next(self.parameters()).device
        present_img_keys = [key for key in self.config.image_features if key in batch]

        # 检查是否有多时间步
        first_key = present_img_keys[0] if present_img_keys else None
        if first_key is None or batch[first_key].dim() != 5:
            return None

        future_images = []
        future_img_masks = []

        for key in present_img_keys:
            img = batch[key]
            # [B, T, C, H, W] → 取 t=1 → [B, C, H, W]
            img = img[:, 1]

            # 检查 padding: obs_{t+k} 可能越界（episode 末尾）
            pad_key = f"{key}_is_pad"
            if pad_key in batch:
                is_pad = batch[pad_key][:, 1]  # [B], True = 越界
            else:
                is_pad = torch.zeros(img.shape[0], dtype=torch.bool, device=device)

            img = self._preprocess_single_image(img, device)
            future_images.append(img)

            # 有效帧 mask：非 padding 的为 True
            mask = (~is_pad).to(device)
            future_img_masks.append(mask)

        # 如果有缺失相机，补 padding
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        for _ in range(len(missing_img_keys)):
            img = torch.ones_like(future_images[-1]) * -1
            mask = torch.zeros(img.shape[0], dtype=torch.bool, device=device)
            future_images.append(img)
            future_img_masks.append(mask)

        return future_images, future_img_masks

    def _preprocess_neg_future_images(
        self, batch: dict[str, Tensor]
    ) -> tuple[list[Tensor], list[Tensor], Tensor] | None:
        """提取 L2 负样本：同 episode 远离 t+k 的 GT obs（CMP L2 负样本）。

        Dataset 已在 __getitem__ 中采样 neg_future_{video_key}，格式 [B, C, H, W]。
        此方法将其处理为模型输入格式。

        Returns:
            (neg_future_images, neg_future_img_masks, neg_future_valid) or None
            - neg_future_images: list of [B, C, H, W]
            - neg_future_img_masks: list of [B] bool
            - neg_future_valid: [B] bool, 标记每个样本 neg 是否有效
        """
        device = next(self.parameters()).device
        present_img_keys = [key for key in self.config.image_features if key in batch]

        # 检查是否有 neg_future 数据
        neg_future_keys = [f"neg_future_{key}" for key in present_img_keys]
        if not any(key in batch for key in neg_future_keys):
            return None

        neg_future_images = []
        neg_future_img_masks = []

        for key in present_img_keys:
            neg_key = f"neg_future_{key}"
            if neg_key in batch:
                img = batch[neg_key]  # [B, C, H, W]
                img = self._preprocess_single_image(img, device)
                neg_future_images.append(img)

                # mask: 所有 neg_future 共享一个 is_valid 标记
                if "neg_future_is_valid" in batch:
                    is_valid = batch["neg_future_is_valid"]  # [B] bool
                else:
                    is_valid = torch.ones(img.shape[0], dtype=torch.bool, device=device)
                neg_future_img_masks.append(is_valid)
            else:
                # 如果某个视角缺失，创建 dummy
                bsize = batch[present_img_keys[0]].shape[0] if present_img_keys else 1
                img = torch.ones(bsize, 3, 224, 224, device=device) * -1
                mask = torch.zeros(bsize, dtype=torch.bool, device=device)
                neg_future_images.append(img)
                neg_future_img_masks.append(mask)

        # 提取全局 valid 标记
        neg_future_valid = batch.get(
            "neg_future_is_valid",
            torch.ones(batch[present_img_keys[0]].shape[0], dtype=torch.bool, device=device)
        )
        return neg_future_images, neg_future_img_masks, neg_future_valid

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        replan_drift_threshold > 0 时：drift 超过阈值触发 replan；
        replan_drift_threshold == 0 时：纯收集模式，用满旧 chunk 全部步。
        """
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        self._select_action_count += 1
        n_action_steps = self.config.n_action_steps
        n_segments = self.config.chunk_size // n_action_steps  # 50 // 10 = 5
        drift_threshold = getattr(self.config, 'replan_drift_threshold', 0.0)
        drift_threshold_mid = getattr(self.config, 'replan_drift_threshold_mid', 0.0)
        replan_mode = getattr(self.config, 'replan_mode', 'drift')

        # ── fixed 模式：每 n_action_steps 步固定 replan（等价 baseline 行为） ──
        if replan_mode == "fixed":
            if len(self._action_queue) == 0:
                self._current_chunk_idx += 1
                actions = self.predict_action_chunk(batch)
                segment = actions[:, :n_action_steps]
                self._action_queue.extend(segment.transpose(0, 1))
                if self._current_chunk_idx <= 2 or self._current_chunk_idx % 5 == 0:
                    print(f"[Fixed-replan] chunk={self._current_chunk_idx}")
            return self._action_queue.popleft()

        # ── drift / 收集模式（原有逻辑不变） ──────────────────────────
        if len(self._action_queue) == 0:
            has_itm = (hasattr(self.model, 'itm_head') and self.model.itm_head is not None
                       and hasattr(self.model, 'obs_query_bridge') and self.model.obs_query_bridge is not None)
            trigger_replan = False  # drift 触发 replan 标志

            # 检查上一段设置的延迟 replan 标志
            if self._force_replan_next:
                trigger_replan = True
                self._force_replan_next = False
                self._replan_count += 1
                print(f"[Replan-deferred] chunk={self._current_chunk_idx}, "
                      f"seg={self._chunk_segment_index}, replan #{self._replan_count}")
                # 预计算 prefix_embs 供 predict_action_chunk 复用
                if has_itm:
                    images, img_masks = self._preprocess_images(batch)
                    tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
                    masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
                    self._cached_prefix_embs, self._cached_prefix_pad_masks, self._cached_prefix_att_masks = \
                        self.model.embed_prefix(images, img_masks, tokens, masks)
            elif self._full_action_chunk is not None and self._chunk_segment_index < n_segments:
                # ---- 旧 chunk 还有剩余段：只算 z1(actual_z3) 做 drift 检测 ----
                seg_idx = self._chunk_segment_index
                start = seg_idx * n_action_steps
                end = start + n_action_steps
                segment = self._full_action_chunk[:, start:end]

                if has_itm:
                    images, img_masks = self._preprocess_images(batch)
                    tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
                    masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
                    attn_act_len = self.config.attn_act_len
                    _, stale_score, actual_z3 = self._should_replan(
                        images, img_masks, tokens, masks, segment[:, :attn_act_len, :],
                    )
                    self._stale_itm_scores.append(stale_score.detach().cpu())
                    self._stale_itm_chunk_ids.append(self._current_chunk_idx)
                    self._stale_itm_segment_ids.append(seg_idx)
                    if len(self._stale_itm_scores) % 10 == 1:
                        print(f"[ITM-stale] chunk={self._current_chunk_idx}, seg={seg_idx}, "
                              f"score={stale_score.mean().item():.4f}")

                    # ForwardModel drift 计算（cosine distance，与训练 loss 一致）
                    if (self._predicted_z3 is not None and actual_z3 is not None
                            and hasattr(self.model, 'forward_model') and self.model.forward_model is not None):
                        drift = 1.0 - F.cosine_similarity(
                            self._predicted_z3.to(actual_z3.device), actual_z3, dim=-1
                        )
                        self._drift_scores.append(drift.detach().cpu())
                        self._drift_chunk_ids.append(self._current_chunk_idx)
                        self._drift_segment_ids.append(seg_idx)

                        # drift-based replan 判断（三级响应）
                        drift_val = drift.mean().item()
                        if drift_threshold > 0 and drift_val > drift_threshold:
                            # 大 drift → 立即 replan
                            trigger_replan = True
                            self._replan_count += 1
                            print(f"[Replan-immediate] chunk={self._current_chunk_idx}, seg={seg_idx}, "
                                  f"drift={drift_val:.4f} > high={drift_threshold:.4f}, "
                                  f"replan #{self._replan_count}")
                        elif drift_threshold_mid > 0 and drift_val > drift_threshold_mid:
                            # 中 drift → 用完当前段，下次强制 replan
                            self._force_replan_next = True
                            print(f"[Replan-scheduled] chunk={self._current_chunk_idx}, seg={seg_idx}, "
                                  f"drift={drift_val:.4f} > mid={drift_threshold_mid:.4f}")
                        if not trigger_replan:
                            # 不 replan 时，为下一段准备 predicted_z3_next
                            if seg_idx + 1 < n_segments:
                                next_start = (seg_idx + 1) * n_action_steps
                                next_end = next_start + n_action_steps
                                next_segment = self._full_action_chunk[:, next_start:next_end]
                                origin_action_dim = self.config.output_features[ACTION].shape[0]
                                z2_next_raw = self.model.content_attention(next_segment[:, :attn_act_len, :origin_action_dim])
                                z2_next = _clamp_embedding_norm(self.model.proj_z2(z2_next_raw).to(dtype=torch.float32), self.config.embedding_max_norm)
                                self._predicted_z3 = self.model.forward_model(actual_z3, z2_next).detach()

                        if len(self._drift_scores) % 10 == 1:
                            print(f"[Drift] chunk={self._current_chunk_idx}, seg={seg_idx}, "
                                  f"drift={drift.mean().item():.4f}")

                if not trigger_replan:
                    # 正常使用旧 segment
                    self._action_queue.extend(segment.transpose(0, 1))
                    self._chunk_segment_index += 1
                # trigger_replan=True 时 fall through 到下面生成新 chunk

            if trigger_replan or self._full_action_chunk is None or self._chunk_segment_index >= n_segments:
                # ---- 需要新 chunk：生成动作，计算 fresh ITM ----
                self._current_chunk_idx += 1
                actions = self.predict_action_chunk(batch)  # 内部计算 fresh ITM + ForwardModel predicted_z3
                self._full_action_chunk = actions.detach().clone()

                # 取第一段放入 queue
                segment = actions[:, :n_action_steps]
                self._action_queue.extend(segment.transpose(0, 1))
                self._chunk_segment_index = 1  # 下次用 segment 1 (stale)

                if self._current_chunk_idx <= 2 or self._current_chunk_idx % 5 == 0:
                    print(f"[ITM-fresh] chunk={self._current_chunk_idx}, "
                          f"fresh_count={len(self._itm_scores)}, stale_count={len(self._stale_itm_scores)}, "
                          f"replan_count={self._replan_count}")

        return self._action_queue.popleft()

    @torch.no_grad()
    def _encode_obs(self, images, img_masks, tokens, masks) -> Tensor:
        """将 obs 编码为 z [B, 512]，与训练时 encode_obs 路径一致。"""
        device = tokens.device
        batch_size = tokens.shape[0]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        cls_head = self.model.cls_head_prefix.expand(batch_size, -1, -1).to(dtype=prefix_embs.dtype)
        prefix_embs_with_cls = torch.cat([cls_head, prefix_embs], dim=1)

        cls_pad_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        prefix_pad_masks_with_cls = torch.cat([cls_pad_mask, prefix_pad_masks], dim=1)
        cls_att_mask = torch.tensor([0], dtype=torch.bool, device=device)
        prefix_att_masks_with_cls = torch.cat(
            [cls_att_mask.unsqueeze(0).expand(batch_size, -1), prefix_att_masks], dim=1
        )

        action_expert_config = get_gemma_config(self.config.action_expert_variant)
        dummy_suffix_embs = torch.zeros(
            batch_size, 1, action_expert_config.width,
            dtype=prefix_embs_with_cls.dtype, device=device,
        )
        dummy_suffix_pad_masks = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        dummy_suffix_att_masks = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        pad_masks = torch.cat([prefix_pad_masks_with_cls, dummy_suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks_with_cls, dummy_suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self.model._prepare_attention_masks_4d(att_2d_masks)

        intermediate_embeds, _ = self.model.paligemma_with_expert.forward_partial(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            inputs_embeds=[prefix_embs_with_cls, None],
            use_cache=False,
            adarms_cond=[None, None],
            part_layer_num=self.model.part_layer_num,
        )

        vlm_seq = intermediate_embeds[0]  # [batch_size, seq_len, 2048]
        z1 = self.model.obs_query_bridge(vlm_seq).to(dtype=torch.float32)  # [batch_size, 512]
        return _clamp_embedding_norm(z1, self.config.embedding_max_norm)

    def _should_replan(
            self,
            images,
            img_masks,
            tokens,
            masks,
            predict_actions: Tensor,
            z1: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Check if replanning is needed based on ITM Head matching score.

        使用训练时相同的编码路径：ObsQueryBridge 提取 z1，ContentAttention + proj_z2 提取 z2，
        ITM Head 判断 (obs, action) 是否匹配。匹配分数低于阈值时触发 replan。

        Args:
            images: Current images (list of tensors)
            img_masks: Current image masks (list of tensors)
            tokens: Current language tokens [batch_size, seq_len]
            masks: Current language token masks [batch_size, seq_len]
            predict_actions: Previously predicted actions [batch_size, seq_len, action_dim]
            z1: 预计算的 obs 嵌入 [batch_size, 512]，如果为 None 则内部编码

        Returns:
            tuple: (should_replan: Tensor [batch_size], itm_score: Tensor [batch_size], z1: Tensor [batch_size, 512])
                   itm_score 是 sigmoid 后的匹配概率，越高表示越匹配
        """
        delta_replan = getattr(self.config, "attn_act_len", 0)
        device = tokens.device
        batch_size = tokens.shape[0]
        no_replan = (
            torch.zeros(batch_size, dtype=torch.bool, device=device),
            torch.ones(batch_size, dtype=torch.float32, device=device),
            None,
        )

        if delta_replan <= 0:
            return no_replan
        if self.model.cls_head_prefix is None or self.model.part_layer_num is None:
            return no_replan
        if self.model.content_attention is None:
            return no_replan
        if not hasattr(self.model, 'obs_query_bridge') or self.model.obs_query_bridge is None:
            return no_replan
        if not hasattr(self.model, 'itm_head') or self.model.itm_head is None:
            return no_replan

        # ========== z1: obs_t → VLM 前 6 层 → ObsQueryBridge ==========
        if z1 is None:
            z1 = self._encode_obs(images, img_masks, tokens, masks)

        # ========== z2: action → ContentAttention → proj_z2 ==========
        attn_act_len = self.model.content_attention.attn_act_len
        origin_action_dim = self.config.output_features[ACTION].shape[0]
        selected_actions = predict_actions[:, :attn_act_len, :origin_action_dim]
        z2_raw = self.model.content_attention(selected_actions)
        z2 = _clamp_embedding_norm(self.model.proj_z2(z2_raw).to(dtype=torch.float32), self.config.embedding_max_norm)  # [batch_size, 512]

        # ========== ITM Head 打分 ==========
        itm_logits = self.model.itm_head(z1, z2)  # [batch_size] raw logits
        itm_score = torch.sigmoid(itm_logits)      # [batch_size] 匹配概率，越高越匹配

        # 分数低于阈值 → 观测与动作不匹配 → 需要 replan
        threshold = getattr(self, '_replan_threshold', 0.5)
        should_replan = itm_score < threshold  # [batch_size] bool
        return should_replan, itm_score, z1

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Sample actions using the model
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        # 旁路计算 ITM score 用于离线分析（不影响动作生成）
        if (hasattr(self.model, 'itm_head') and self.model.itm_head is not None
                and hasattr(self.model, 'obs_query_bridge') and self.model.obs_query_bridge is not None):
            attn_act_len = self.config.attn_act_len
            _, itm_score, z1 = self._should_replan(
                images, img_masks, tokens, masks, actions[:, :attn_act_len, :],
            )
            self._itm_scores.append(itm_score.detach().cpu())
            # 每 10 次打印一次 ITM score 信息
            if len(self._itm_scores) % 10 == 1:
                print(f"[ITM] chunk #{len(self._itm_scores)}, score: {itm_score.mean().item():.4f} (batch mean)")

            # 计算 z2 用于模态对齐分析和 ForwardModel（当 z1 有效时）
            if z1 is not None:
                origin_action_dim = self.config.output_features[ACTION].shape[0]
                z2_raw = self.model.content_attention(actions[:, :attn_act_len, :origin_action_dim])
                z2 = _clamp_embedding_norm(
                    self.model.proj_z2(z2_raw).to(dtype=torch.float32),
                    self.config.embedding_max_norm,
                )
                # 存储 z1/z2 嵌入用于模态对齐分析
                self._z1_embeddings.append(z1.detach().cpu())
                self._z2_embeddings.append(z2.detach().cpu())

                # ForwardModel: 预测执行 action[0:k] 后的 obs 嵌入
                if hasattr(self.model, 'forward_model') and self.model.forward_model is not None:
                    self._predicted_z3 = self.model.forward_model(z1, z2).detach()

        return actions

    def forward(self, batch: dict[str, Tensor], online=False, bc_only=False,
                feature_weight: float = 0.0,
                feature_batch: dict[str, Tensor] | None = None) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: 输入数据批次
            online: 是否为在线训练模式（feature_loss only）
            bc_only: 是否只计算 bc_loss（跳过 CMP），用于在线 BC 训练
            feature_weight: 联合 feature_loss 权重。>0 时在 bc_only 路径中追加
                           feature_loss: total = bc_loss + feature_weight * feature_loss。
            feature_batch: 用于 feature_loss 的在线数据批次（包含 prev_actions, pred_action,
                          以及对应的 images/tokens）。如果为 None 且 feature_weight > 0，
                          则从 batch 中取 prev_actions/pred_action（向后兼容）。
        """
        # 功能总结：根据 online 标志计算在线 loss 或离线 BC+CMP 组合 loss，并返回日志字典。
        original_action_dim = self.config.output_features[ACTION].shape[0]
        if online:
            # 在线训练：仅更新与动作生成相关的子模块
            if self.offline_mode is True:
                self._apply_online_params()

        if online:
            prev_actions = batch.get('prev_actions')  # [batch, 10, action_dim]
            pred_action = batch.get('pred_action')  # [batch, 10, action_dim]

            if prev_actions is None or pred_action is None:
                zero_loss = torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
                return zero_loss, {"feature_loss": 0.0, "l2_actions": 0.0, "skipped": True}

            # 准备输入
            images, img_masks = self._preprocess_images(batch)
            tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

            # 1) prev_actions 经过 content_attention 得到条件 token
            condition_token = self.model.content_attention(prev_actions)  # [batch, hidden_dim]

            # 2. 使用 condition_token 生成动作（可微分）
            generated_actions = self.model.sample_actions_differentiable(
                images, img_masks, tokens, masks,
                condition_token=condition_token,
                num_steps=self.config.num_inference_steps,
            )

            # 3. 生成动作的前 attn_act_len 步过 content_attention
            attn_act_len = self.model.content_attention.attn_act_len
            gen_actions_slice = generated_actions[:, :attn_act_len, :original_action_dim]  # [batch, attn_act_len, action_dim]
            gen_feature = self.model.content_attention(gen_actions_slice)  # [batch, hidden_dim]

            # 4. pred_action 过 content_attention
            pred_feature = self.model.content_attention(pred_action)  # [batch, hidden_dim]

            # 5. 计算 per-sample 特征距离 loss
            per_sample_feature_loss = F.relu(torch.square(gen_feature - pred_feature) - 0.1).mean(dim=-1)  # [batch]

            # 6. Success 加权：成功帧=1.0，失败帧按轨迹位置指数衰减
            episode_success = batch.get("episode_success")  # [batch, 1] or None
            if episode_success is not None:
                success = episode_success.squeeze(-1)  # [batch]
                frame_idx = batch.get("frame_index")  # [batch, 1]
                if frame_idx is not None:
                    t_normalized = frame_idx.squeeze(-1).float() / 520.0  # 归一化位置
                    decay_rate = getattr(self.config, 'online_success_decay_rate', 8.0)
                    fail_weight = torch.exp(-decay_rate * t_normalized)
                    weight = torch.where(success > 0.5, torch.ones_like(success), fail_weight)
                else:
                    weight = torch.where(success > 0.5, torch.ones_like(success),
                                         torch.full_like(success, 0.3))
                feature_loss = (weight * per_sample_feature_loss).sum() / weight.sum().clamp(min=1)
            else:
                # 兼容旧数据集：无 episode_success → 原逻辑
                weight = None
                feature_loss = per_sample_feature_loss.mean()

            # 7. 计算 generated_actions 与 pred_action 在前 attn_act_len 步的 L2 距离
            pred_actions_slice = pred_action[:, :attn_act_len, :original_action_dim]
            l2_per_step = torch.mean(torch.square(gen_actions_slice - pred_actions_slice), dim=-1)  # [batch, attn_act_len]
            l2_actions = l2_per_step.mean()

            loss_dict = {
                "feature_loss": feature_loss.item(),
                "l2_actions": l2_actions.item(),
                "avg_weight": weight.mean().item() if weight is not None else 1.0,
                "n_success": int((episode_success.squeeze(-1) > 0.5).sum().item()) if episode_success is not None else -1,
            }

            return feature_loss, loss_dict

        else:
            # Prepare inputs
            images, img_masks = self._preprocess_images(batch)
            tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

            # CMP 相关预处理：bc_only 不需要 future/neg 数据，跳过节省计算
            if not bc_only:
                # 提取未来帧（三元因果链用）
                future_result = self._preprocess_future_images(batch)
                future_images = future_result[0] if future_result is not None else None
                future_img_masks = future_result[1] if future_result is not None else None

            # 提取同 episode GT 时序错位负样本（CMP L1 用）
            neg_gt_actions = batch.get("neg_action", None)
            neg_action_valid = batch.get("neg_action_is_valid", None)
            neg_action_distance = batch.get("neg_action_distance", None)

            # Debug: 检查 batch 中是否包含负样本 key
            if not hasattr(self, '_batch_debug_printed'):
                self._batch_debug_printed = True
                neg_keys = [k for k in batch.keys() if 'neg' in k.lower()]
                all_keys = sorted(batch.keys())
                print(f"[forward DEBUG] batch keys: {all_keys}")
                print(f"[forward DEBUG] neg-related keys: {neg_keys}")
                print(f"[forward DEBUG] neg_gt_actions is None: {neg_gt_actions is None}")
                print(f"[forward DEBUG] neg_action_valid is None: {neg_action_valid is None}")

            # 提取同 episode 远离 t+k 的 GT obs（CMP L2 用）
            neg_future_result = self._preprocess_neg_future_images(batch)
            if neg_future_result is not None:
                neg_future_images, neg_future_img_masks, neg_future_valid = neg_future_result
            else:
                neg_future_images, neg_future_img_masks, neg_future_valid = None, None, None

            if self.config.cmp_pretrain:
                # CMP 预训练：仅使用 CMP loss（含前向动力学项）
                if self.offline_mode is False:
                    self._apply_param_cmp()

                cmp_losses, cmp_loss_info = self.model.forward_cmp(
                    images, img_masks, tokens, masks,
                    future_images=future_images, future_img_masks=future_img_masks,
                    neg_gt_actions=neg_gt_actions, neg_action_valid=neg_action_valid,
                    neg_future_images=neg_future_images, neg_future_img_masks=neg_future_img_masks,
                    neg_future_valid=neg_future_valid,
                    neg_action_distance=neg_action_distance,
                )
                # CMP loss 缩放（与外部实验设置一致）
                losses = 0.01 * cmp_losses.mean()

                loss_dict = {
                    "loss": losses.item(),
                    "bc_loss": 0.,
                    "cmp_loss": losses.item(),
                }
                loss_dict.update(cmp_loss_info)
            else:
                # 离线联合训练：BC loss + CMP loss
                if self.offline_mode is False:
                    self._apply_offline_params()
                # 1) BC：动作去噪回归损失
                actions = self.prepare_action(batch)
                anchor_weight = getattr(self.config, 'anchor_weight', 0.0)
                need_anchor = (bc_only and anchor_weight > 0
                               and self.model._anchor_expert_state is not None)
                if need_anchor:
                    bc_losses, _internals = self.model.forward(
                        images, img_masks, tokens, masks, actions,
                        return_internals=True)
                else:
                    bc_losses = self.model.forward(images, img_masks, tokens, masks, actions)
                bc_loss = bc_losses[:, :, :original_action_dim].mean()

                if not bc_only:
                    # 2) CMP：三元因果链对比损失 + 前向动力学项
                    cmp_losses, cmp_loss_info = self.model.forward_cmp(
                        images, img_masks, tokens, masks,
                        future_images=future_images, future_img_masks=future_img_masks,
                        neg_gt_actions=neg_gt_actions, neg_action_valid=neg_action_valid,
                        neg_future_images=neg_future_images, neg_future_img_masks=neg_future_img_masks,
                        neg_future_valid=neg_future_valid,
                        neg_action_distance=neg_action_distance,
                    )
                    cmp_loss = 0.1 * cmp_losses.mean()
                    # 3) 总损失：BC + 缩放后的 CMP
                    losses = bc_loss + cmp_loss

                    loss_dict = {
                        "loss": losses.item(),
                        "bc_loss": bc_loss.item(),
                        "cmp_loss": cmp_loss.item(),
                    }
                    loss_dict.update(cmp_loss_info)
                else:
                    # 在线 BC 训练：bc_loss，按 episode 成功率加权（如果有）
                    # 成功帧权重=1.0，失败帧按轨迹位置衰减（早期帧权重高，后期帧权重低）
                    feature_only_online = getattr(self.config, 'feature_only_online', False)

                    per_sample_bc = bc_losses[:, :, :original_action_dim].mean(dim=(1, 2))  # [batch]

                    episode_success = batch.get("episode_success")
                    if episode_success is not None:
                        success = episode_success.squeeze(-1)  # [batch]
                        frame_idx = batch.get("frame_index")  # [batch, 1] or [batch]
                        if frame_idx is not None:
                            t_normalized = frame_idx.squeeze(-1).float() / 520.0
                            decay_rate = getattr(self.config, 'online_success_decay_rate', 8.0)
                            fail_weight = torch.exp(-decay_rate * t_normalized)
                            weight = torch.where(success > 0.5, torch.ones_like(success), fail_weight)
                        else:
                            weight = torch.where(success > 0.5, torch.ones_like(success),
                                                 torch.full_like(success, 0.3))
                        bc_loss = (weight * per_sample_bc).sum() / weight.sum().clamp(min=1)
                        avg_weight = weight.mean().item()
                        n_success = int((success > 0.5).sum().item())
                    else:
                        # 离线数据没有 episode_success → 无权重，直接 mean
                        bc_loss = per_sample_bc.mean()
                        avg_weight = 1.0
                        n_success = -1

                    # Feature-only mode: zero out bc_loss (still logged for diagnostics)
                    if feature_only_online:
                        losses = torch.tensor(0.0, device=bc_loss.device, dtype=bc_loss.dtype)
                    else:
                        losses = bc_loss

                    loss_dict = {
                        "loss": losses.item(),
                        "bc_loss": bc_loss.item(),
                        "cmp_loss": 0.0,
                        "avg_weight": avg_weight,
                        "n_success": n_success,
                    }

                    # ====== Anchor loss (velocity distillation) ======
                    if need_anchor:
                        v_anchor = self.model._forward_anchor(_internals)
                        v_current = _internals["v_t"]
                        anchor_loss = F.mse_loss(
                            v_current[:, :, :original_action_dim],
                            v_anchor[:, :, :original_action_dim],
                        )
                        losses = losses + anchor_weight * anchor_loss
                        loss_dict["loss"] = losses.item()
                        loss_dict["anchor_loss"] = anchor_loss.item()
                    else:
                        loss_dict["anchor_loss"] = 0.0

                    # ====== 联合 feature_loss（bc_only + feature_weight > 0）======
                    # 使用 feature_batch（在线数据）的 images 和 prev_actions/pred_action
                    # 计算 feature_loss，与 bc_loss 联合优化
                    if feature_weight > 0:
                        fb = feature_batch if feature_batch is not None else batch
                        prev_actions = fb.get('prev_actions')
                        pred_action = fb.get('pred_action')
                        if prev_actions is not None and pred_action is not None:
                            # 用 feature_batch 的 images/tokens（在线 rollout 观测）
                            feat_images, feat_img_masks = self._preprocess_images(fb)
                            feat_tokens = fb[f"{OBS_LANGUAGE_TOKENS}"]
                            feat_masks = fb[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

                            # condition_token: prev_actions → content_attention（冻结前向）
                            condition_token = self.model.content_attention(prev_actions)

                            # 可微分动作生成（N 步 ODE），用在线观测生成动作
                            generated_actions = self.model.sample_actions_differentiable(
                                feat_images, feat_img_masks, feat_tokens, feat_masks,
                                condition_token=condition_token,
                                num_steps=self.config.num_inference_steps,
                            )

                            # gen_feature / pred_feature：content_attention（冻结前向）
                            attn_act_len = self.model.content_attention.attn_act_len
                            gen_actions_slice = generated_actions[:, :attn_act_len, :original_action_dim]
                            gen_feature = self.model.content_attention(gen_actions_slice)
                            pred_feature = self.model.content_attention(pred_action)

                            # per-sample feature loss with margin
                            per_sample_fl = F.relu(
                                torch.square(gen_feature - pred_feature) - 0.1
                            ).mean(dim=-1)  # [batch]

                            # feature_loss weighting
                            feat_episode_success = fb.get("episode_success")
                            if feature_only_online:
                                # 硬过滤：只用成功 episode（sampler 已过滤，这里是安全检查）
                                if feat_episode_success is not None:
                                    success_mask = feat_episode_success.squeeze(-1) > 0.5
                                    if success_mask.any():
                                        feature_loss = per_sample_fl[success_mask].mean()
                                    else:
                                        feature_loss = per_sample_fl.mean() * 0.0  # 保持 autograd graph
                                else:
                                    feature_loss = per_sample_fl.mean()
                            elif feat_episode_success is not None:
                                # 原来的软降权逻辑
                                feat_success = feat_episode_success.squeeze(-1)
                                feat_frame_idx = fb.get("frame_index")
                                if feat_frame_idx is not None:
                                    ft_norm = feat_frame_idx.squeeze(-1).float() / 520.0
                                    fdecay = getattr(self.config, 'online_success_decay_rate', 8.0)
                                    feat_fail_w = torch.exp(-fdecay * ft_norm)
                                    feat_weight = torch.where(
                                        feat_success > 0.5,
                                        torch.ones_like(feat_success), feat_fail_w)
                                else:
                                    feat_weight = torch.where(
                                        feat_success > 0.5,
                                        torch.ones_like(feat_success),
                                        torch.full_like(feat_success, 0.3))
                                feature_loss = (feat_weight * per_sample_fl).sum() / feat_weight.sum().clamp(min=1)
                            else:
                                feature_loss = per_sample_fl.mean()

                            # L2 action distance 诊断
                            pred_actions_slice = pred_action[:, :attn_act_len, :original_action_dim]
                            l2_actions = torch.mean(
                                torch.square(gen_actions_slice - pred_actions_slice)
                            )

                            # 联合损失（losses 已包含 bc_loss + anchor_loss，不能覆盖）
                            losses = losses + feature_weight * feature_loss

                            loss_dict["loss"] = losses.item()
                            loss_dict["feature_loss"] = feature_loss.item()
                            loss_dict["l2_actions"] = l2_actions.item()
                        else:
                            # batch 中没有 prev_actions/pred_action → 只用 bc_loss
                            loss_dict["feature_loss"] = 0.0
                            loss_dict["l2_actions"] = 0.0

            return losses, loss_dict

    def get_action_context(self, batch_size: int = 1, prev_steps: int | None = None,
                           pred_steps: int | None = None) -> dict:
        """获取当前步骤的动作上下文信息。

        用于在线数据收集时记录每帧的动作上下文：
        - prev_actions: 前 prev_steps 步已执行的动作
        - pred_action: 后 pred_steps 步预测的动作
        - actions_seq_valid: 仅当前后动作都来自同一次预测时为 True

        Args:
            batch_size: batch 大小（默认1）
            prev_steps: 需要的前序动作步数（默认从 config.attn_act_len 读取）
            pred_steps: 需要的预测动作步数（默认从 config.attn_act_len 读取）

        Returns:
            dict: {
                'prev_actions': Tensor [batch, prev_steps, action_dim],
                'pred_action': Tensor [batch, pred_steps, action_dim],
                'actions_seq_valid': Tensor [batch] bool,
            }
            注意：始终返回正确形状的数据，无效时用零填充且 valid=False
        """
        if prev_steps is None:
            prev_steps = self.config.attn_act_len
        if pred_steps is None:
            pred_steps = self.config.attn_act_len

        action_dim = self.config.output_features[ACTION].shape[0]
        device = next(self.parameters()).device

        # 默认返回零填充
        prev_actions = torch.zeros(batch_size, prev_steps, action_dim, device=device, dtype=torch.float32)
        pred_action = torch.zeros(batch_size, pred_steps, action_dim, device=device, dtype=torch.float32)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if self._full_action_chunk is None:
            return {'prev_actions': prev_actions, 'pred_action': pred_action, 'actions_seq_valid': valid_mask}

        # 从现有状态推算当前在 chunk 中的位置
        # _chunk_segment_index: 下一段的索引（1 表示 segment 0 已加载）
        # _action_queue: 当前段中尚未消耗的动作
        n_act = self.config.n_action_steps
        queue_len = len(self._action_queue)
        if queue_len > 0:
            # 正在消耗某段中间：已完成段数 * n_act + 当前段已消耗的动作数
            pos = (self._chunk_segment_index - 1) * n_act + (n_act - queue_len)
        else:
            # 当前段刚好用完，下一次 select_action 会加载新段
            pos = self._chunk_segment_index * n_act

        chunk_size = self.config.chunk_size
        if pos >= prev_steps and pos + pred_steps <= chunk_size:
            chunk = self._full_action_chunk[:, :, :action_dim]
            prev_actions = chunk[:, pos - prev_steps : pos, :]
            pred_action = chunk[:, pos : pos + pred_steps, :]
            valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        return {'prev_actions': prev_actions, 'pred_action': pred_action, 'actions_seq_valid': valid_mask}
