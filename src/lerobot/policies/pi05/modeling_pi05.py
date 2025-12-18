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

def vicreg_loss(
    z1: Tensor,
    z2: Tensor,
    lambda_param: float = 25.0,
    mu_param: float = 25.0,
    nu_param: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Tensor:
    """VICReg loss with Variance-Invariance-Covariance Regularization (PyTorch version).

    Args:
        z1: First representation [batch, num_tokens, dim] or [batch, dim]
        z2: Second representation [batch, num_tokens, dim] or [batch, dim]
        lambda_param: Weight for invariance loss (default: 25.0)
        mu_param: Weight for variance loss (default: 25.0)
        nu_param: Weight for covariance loss (default: 1.0)
        gamma: Target standard deviation (default: 1.0)
        eps: Small constant for numerical stability

    Returns:
        VICReg loss value (scalar tensor)
    """
    # Handle both 2D and 3D inputs
    if z1.dim() == 2:
        z1 = z1.unsqueeze(1)  # [batch, dim] -> [batch, 1, dim]
    if z2.dim() == 2:
        z2 = z2.unsqueeze(1)  # [batch, dim] -> [batch, 1, dim]

    batch_size, num_tokens, dim = z1.shape

    # Reshape to (batch*num_tokens, dim)
    z1_flat = z1.reshape(-1, dim)  # [batch*num_tokens, dim]
    z2_flat = z2.reshape(-1, dim)  # [batch*num_tokens, dim]
    n_samples = z1_flat.shape[0]

    # Invariance loss: L2 distance between corresponding representations
    invariance_loss = torch.mean(torch.square(z1_flat - z2_flat), dim=-1)  # [batch*num_tokens]
    invariance_loss = torch.mean(invariance_loss)  # scalar

    # Variance loss: encourage standard deviation to be close to gamma
    std_z1 = torch.sqrt(torch.var(z1_flat, dim=0) + eps)  # [dim]
    std_z2 = torch.sqrt(torch.var(z2_flat, dim=0) + eps)  # [dim]
    variance_loss = torch.mean(F.relu(gamma - std_z1)) + torch.mean(F.relu(gamma - std_z2))

    # Covariance loss: encourage decorrelation of features
    z1_centered = z1_flat - torch.mean(z1_flat, dim=0, keepdim=True)  # [batch*num_tokens, dim]
    z2_centered = z2_flat - torch.mean(z2_flat, dim=0, keepdim=True)  # [batch*num_tokens, dim]

    cov_z1 = (z1_centered.T @ z1_centered) / (n_samples - 1)  # [dim, dim]
    cov_z2 = (z2_centered.T @ z2_centered) / (n_samples - 1)  # [dim, dim]

    # Off-diagonal mask
    off_diagonal_mask = 1 - torch.eye(dim, device=z1.device, dtype=z1.dtype)
    offdiag_z1 = cov_z1 * off_diagonal_mask
    offdiag_z2 = cov_z2 * off_diagonal_mask

    # Covariance loss (normalize by dim, not number of elements)
    cov_loss_z1 = torch.sum(torch.square(offdiag_z1)) / dim
    cov_loss_z2 = torch.sum(torch.square(offdiag_z2)) / dim
    covariance_loss = cov_loss_z1 + cov_loss_z2

    # Total loss
    total_loss = lambda_param * invariance_loss + mu_param * variance_loss + nu_param * covariance_loss

    print("----------------")
    print(lambda_param * invariance_loss)
    print(mu_param * variance_loss)
    print(nu_param * covariance_loss)

    return total_loss

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
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

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
                        zero_cond = torch.zeros(batch_size, cond_dim, device=hidden_states.device, dtype=hidden_states.dtype)
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
        ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

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
        #self.layer_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
        x_concat = torch.cat([cls, x], dim=1)  # [batch_size, 1 + attn_act_len, hidden_dim]
        #x_normed = self.layer_norm(x_concat)
        x_normed = x_concat
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
        self.cmp_step=0

        # 投影头：将 paligemma 的 hidden_dim (2048) 投影到 action_expert 的 hidden_dim (1024)
        # 用于对比学习中的维度匹配
        self.cmp_projection = nn.Sequential(
            nn.Linear(paligemma_config.width, paligemma_config.width),
            #nn.LayerNorm(paligemma_config.width),
            nn.GELU(),
            nn.Linear(paligemma_config.width, action_expert_config.width),
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

    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None, condition_token=None) -> Tensor:
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

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    def forward_cmp(
            self,
            images,
            img_masks,
            tokens,
            masks,
            lambda_param: float = 25.0,
            mu_param: float = 25.0,
            nu_param: float = 1.0,
            gamma: float = 1.0,
    ) -> Tensor:
        """对比学习前向传播，使用 VICReg loss。

        Args:
            images: 图像列表
            img_masks: 图像掩码列表
            tokens: 语言 tokens
            masks: 语言 token 掩码
            actions: 动作张量 [batch_size, chunk_size, action_dim]
            lambda_param: VICReg invariance loss 权重
            mu_param: VICReg variance loss 权重
            nu_param: VICReg covariance loss 权重
            gamma: VICReg 目标标准差

        Returns:
            VICReg loss (scalar tensor)
        """
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

        batch_size = tokens.shape[0]
        device = tokens.device

        # 第一步：准备 prefix embeddings，并将 cls_head_prefix 添加到前面
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)

        # 将 cls_head_prefix 添加到 prefix_embs 的前面
        # 确保数据类型匹配
        cls_head = self.cls_head_prefix.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]
        cls_head = cls_head.to(dtype=prefix_embs.dtype)  # 确保数据类型匹配
        prefix_embs_with_cls = torch.cat([cls_head, prefix_embs], dim=1)  # [batch_size, 1 + seq_len, hidden_dim]

        # 更新 pad_masks 和 att_masks，为 cls_head 添加对应的掩码
        cls_pad_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        prefix_pad_masks_with_cls = torch.cat([cls_pad_mask, prefix_pad_masks], dim=1)

        cls_att_mask = torch.tensor([0], dtype=torch.bool, device=device)  # cls_head 可以 attend 到所有之前的内容
        prefix_att_masks_with_cls = torch.cat([cls_att_mask.unsqueeze(0).expand(batch_size, -1), prefix_att_masks],
                                              dim=1)

        # 准备一个 dummy suffix_embs（forward_partial 需要两个输入）
        action_expert_config = get_gemma_config(self.config.action_expert_variant)
        dummy_suffix_embs = torch.zeros(
            batch_size, 1, action_expert_config.width, dtype=prefix_embs_with_cls.dtype, device=device
        )
        dummy_suffix_pad_masks = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        dummy_suffix_att_masks = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        # 合并 masks
        pad_masks = torch.cat([prefix_pad_masks_with_cls, dummy_suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks_with_cls, dummy_suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

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

        intermediate_embeds, intermediate_state = self._apply_checkpoint(
            forward_partial_func, prefix_embs_with_cls, dummy_suffix_embs, att_2d_masks_4d, position_ids
        )

        # 从中间输出中提取分类头对应的输出（第一个 token）
        cmp_vec_0 = intermediate_embeds[0][:, 0, :]  # [batch_size, paligemma_hidden_dim]
        # 通过投影头将 cmp_vec_0 从 paligemma 的维度投影到 action_expert 的维度
        cmp_vec_0 = self.cmp_projection(cmp_vec_0)  # [batch_size, action_expert_hidden_dim]
        cmp_vec_0 = cmp_vec_0.to(dtype=torch.float32)  # 转换为 float32 用于损失计算

        # 确保维度正确
        if cmp_vec_0.dim() != 2:
            raise ValueError(f"Expected cmp_vec_0 to be 2D [batch_size, hidden_dim], got shape {cmp_vec_0.shape}")

        # 第二步：将 actions 输入到 SingleHeadContentAttention 中
        # 首先需要将 actions 投影到 hidden_dim，然后组织成 [batch_size, attn_act_len, hidden_dim] 的格式
        attn_act_len = self.content_attention.attn_act_len

        actions = self.sample_actions(images, img_masks, tokens, masks)
        origin_action_dim = self.config.output_features[ACTION].shape[0]
        # 选择前 attn_act_len 个 action steps
        selected_actions = actions[:, :attn_act_len, :origin_action_dim]  # [batch_size, attn_act_len, max_action_dim]

        # 通过 SingleHeadContentAttention 得到 cmp_vec_1
        cmp_vec_1 = self.content_attention(selected_actions)  # [batch_size, hidden_dim]
        cmp_vec_1 = cmp_vec_1.to(dtype=torch.float32)  # 转换为 float32 用于损失计算

        # 确保维度匹配
        if cmp_vec_0.shape != cmp_vec_1.shape:
            raise ValueError(
                f"Dimension mismatch: cmp_vec_0 shape {cmp_vec_0.shape} != cmp_vec_1 shape {cmp_vec_1.shape}"
            )

        if self.cmp_step < 200:
            lambda_param = 0
            nu_param = 0

        # 第三步：使用 VICReg loss 计算对比学习损失
        loss_scalar = vicreg_loss(
            cmp_vec_0,
            cmp_vec_1,
            lambda_param=lambda_param,
            mu_param=mu_param,
            nu_param=nu_param,
            gamma=gamma,
        )

        # 将标量损失扩展为 [batch_size, 1, action_dim] 以匹配 forward 的返回格式
        # vicreg_loss 返回的是标量（0维），没有 batch 维度，需要扩展
        batch_size = cmp_vec_0.shape[0]
        action_dim = self.config.max_action_dim

        # 将标量损失扩展为 [batch_size, 1, action_dim]
        losses = loss_scalar.view(1, 1, 1).expand(batch_size, 1, action_dim)

        self.cmp_step += 1
        return losses

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
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
        suffix_out = suffix_out[:, -self.config.chunk_size :]
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
            
            suffix_out = suffix_out[:, -self.config.chunk_size :]
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

        self.train_addition_only: bool = False

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
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

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

    def _apply_param_freezing(self) -> None:
        """根据 train_addition_only 冻结或解冻参数。"""
        # 如果不只训练新增参数，则全部参与训练
        if not self.train_addition_only:
            for p in self.parameters():
                p.requires_grad = True
            return

        # 只训练 addition_params，其它参数 requires_grad=False
        params = []

        # content_attention 的参数
        if self.model.content_attention is not None:
            params.extend(self.model.content_attention.parameters())

        if self.model.cmp_projection is not None:
            params.extend(self.model.cmp_projection.parameters())

        # cls_head_prefix 参数
        if self.model.cls_head_prefix is not None:
            params.append(self.model.cls_head_prefix)

        addition_ids = {id(p) for p in params}
        for p in self.parameters():
            p.requires_grad = id(p) in addition_ids

    def _apply_online_params(self) -> None:
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

    def freeze_params(self) -> None:
        """设置是否只训练新增参数，并立即应用到 requires_grad。"""
        self.train_addition_only = True
        self._apply_param_freezing()

    def unfreeze_params(self) -> None:
        """显式解冻所有参数，恢复联合训练。"""
        self.train_addition_only = False
        self._apply_param_freezing()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        
        # 动作上下文追踪：每个样本在当前 chunk 中已执行的步数
        self.step_num_in_chunk: Tensor | None = None
        self._predicted_actions_buffer: Tensor | None = None
        self._last_actions_list: Tensor | None = None

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

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        self.step_num_in_chunk += 1
        return self._action_queue.popleft()

    def _should_replan(
        self,
        images,
        img_masks,
        tokens,
        masks,
        predict_actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Check if replanning is needed based on VICReg similarity comparison.

        Args:
            images: Current images (list of tensors)
            img_masks: Current image masks (list of tensors)
            tokens: Current language tokens [batch_size, seq_len]
            masks: Current language token masks [batch_size, seq_len]
            predict_actions: Previously predicted actions [batch_size, delta_replan, action_dim]

        Returns:
            tuple: (should_replan: Tensor [batch_size], similarity: Tensor [batch_size])
        """
        delta_replan = getattr(self.config, "delta_replan", 0)
        device = tokens.device
        batch_size = tokens.shape[0]
        if delta_replan <= 0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device), torch.zeros(batch_size, dtype=torch.float32, device=device)

        if self.model.cls_head_prefix is None or self.model.part_layer_num is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device), torch.zeros(batch_size, dtype=torch.float32, device=device)

        if self.model.content_attention is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device), torch.zeros(batch_size, dtype=torch.float32, device=device)

        device = tokens.device
        batch_size = tokens.shape[0]

        # Step 1: Get cmp_vec_0 from current visual-language information
        # Prepare prefix embeddings with cls_head_prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )

        cls_head = self.model.cls_head_prefix.expand(batch_size, -1, -1).to(
            dtype=prefix_embs.dtype
        )
        prefix_embs_with_cls = torch.cat([cls_head, prefix_embs], dim=1)

        # Update masks
        cls_pad_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        prefix_pad_masks_with_cls = torch.cat([cls_pad_mask, prefix_pad_masks], dim=1)

        cls_att_mask = torch.tensor([0], dtype=torch.bool, device=device)
        prefix_att_masks_with_cls = torch.cat(
            [cls_att_mask.unsqueeze(0).expand(batch_size, -1), prefix_att_masks], dim=1
        )

        # Prepare dummy suffix
        action_expert_config = get_gemma_config(self.config.action_expert_variant)
        dummy_suffix_embs = torch.zeros(
            batch_size,
            1,
            action_expert_config.width,
            dtype=prefix_embs_with_cls.dtype,
            device=device,
        )
        dummy_suffix_pad_masks = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        dummy_suffix_att_masks = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        # Merge masks
        pad_masks = torch.cat([prefix_pad_masks_with_cls, dummy_suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks_with_cls, dummy_suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self.model._prepare_attention_masks_4d(att_2d_masks)

        # Execute forward_partial
        intermediate_embeds, _ = self.model.paligemma_with_expert.forward_partial(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            inputs_embeds=[prefix_embs_with_cls, None],
            use_cache=False,
            adarms_cond=[None, None],
            part_layer_num=self.model.part_layer_num,
        )

        # Extract cls head output
        cmp_vec_0 = intermediate_embeds[0][:, 0, :]  # [batch_size, paligemma_hidden_dim]
        cmp_vec_0 = self.model.cmp_projection(cmp_vec_0)  # [batch_size, action_expert_hidden_dim]
        cmp_vec_0 = cmp_vec_0.to(dtype=torch.float32)

        # Step 2: Get cmp_vec_1 from predicted actions
        attn_act_len = self.model.content_attention.attn_act_len

        origin_action_dim = self.config.output_features[ACTION].shape[0]
        # Select first attn_act_len actions
        predict_actions = predict_actions[:, :attn_act_len, :origin_action_dim]


        # Get cmp_vec_1
        cmp_vec_1 = self.model.content_attention(predict_actions).to(dtype=torch.float32)

        # Step 3: Compute similarity
        similarity = compute_vicreg_similarity(cmp_vec_0, cmp_vec_1)  # [batch_size]

        # Decide if replanning is needed (higher similarity = more different = need replan)
        should_replan = similarity > self._replan_threshold  # [batch_size] bool tensor
        return should_replan, similarity

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations with replanning support.

        Replanning logic:
        - Initially: directly predict actions
        - After that: every delta_replan steps, compare current observation with predicted actions
        - If similarity > threshold (0.1): replan (generate new actions)
        - If similarity <= threshold: continue using previous predictions
        """
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        delta_replan = self.config.attn_act_len

        # Initialize replanning state if not exists
        if delta_replan > 0:
            if not hasattr(self, "_replan_step_counter"):
                self._replan_step_counter = 0
            if not hasattr(self, "_predicted_actions_buffer"):
                self._predicted_actions_buffer = None
            if not hasattr(self, "_replan_threshold"):
                self._replan_threshold = 0.25

        batch_size = tokens.shape[0]
        device = tokens.device
        
        # Initialize should_replan as all True (default: always plan initially or when buffer is empty)
        should_replan = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Check if we need to replan based on similarity comparison
        if delta_replan > 0 and self._predicted_actions_buffer is not None:
            # Check every delta_replan steps
            should_replan, similarity = self._should_replan(
                images, img_masks, tokens, masks, self._predicted_actions_buffer[:,:delta_replan,:],
            )
            should_replan = should_replan | (self.step_num_in_chunk >= self.config.chunk_size)


        # If replanning is needed, generate new actions
        # Handle list indexing for images and img_masks
        if should_replan.all():
            # All samples need replanning, use all inputs
            filtered_images = images
            filtered_img_masks = img_masks
            filtered_tokens = tokens
            filtered_masks = masks
        elif should_replan.any():
            # Some samples need replanning
            # Filter images and img_masks (they are lists)
            filtered_images = [img[should_replan] for img in images]
            filtered_img_masks = [mask[should_replan] for mask in img_masks]
            filtered_tokens = tokens[should_replan]
            filtered_masks = masks[should_replan]


        # Generate actions only for samples that need replanning
        if should_replan.any():
            actions = self.model.sample_actions(filtered_images, filtered_img_masks, filtered_tokens, filtered_masks, **kwargs)
            # Update buffer for samples that were replanned
            if self._predicted_actions_buffer is not None:
                # print(self._predicted_actions_buffer[should_replan].shape)
                # print(self._predicted_actions_buffer.shape)
                # print(actions.shape)
                # print("22222222")
                self._predicted_actions_buffer[should_replan] = actions

                self.step_num_in_chunk[should_replan] = 0
            else:
                self._predicted_actions_buffer = actions
                self.step_num_in_chunk = torch.zeros(batch_size, dtype=torch.long, device=device)


        original_action_dim = self.config.output_features[ACTION].shape[0]
        ret =  self._predicted_actions_buffer[:, :delta_replan, :original_action_dim]
        self._last_actions_list = self._predicted_actions_buffer[:, :delta_replan, :original_action_dim]

        max_action_len = self._predicted_actions_buffer.shape[-1]
        self._predicted_actions_buffer = torch.cat((self._predicted_actions_buffer[:, delta_replan:, :], torch.zeros((batch_size, delta_replan, max_action_len)).to(device)), dim=1)
        return ret

    def forward(self, batch: dict[str, Tensor], cmp=False, online=False) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.
        
        Args:
            batch: 输入数据批次
            cmp: 是否为对比学习模式
            online: 是否为在线训练模式
                    如果为 True，使用 prev_actions 作为条件，生成动作并计算与 pred_action 的特征距离
        """
        original_action_dim = self.config.output_features[ACTION].shape[0]
        if online:
            self._apply_online_params()
            prev_actions = batch.get('prev_actions')      # [batch, 10, action_dim]
            pred_action = batch.get('pred_action')        # [batch, 10, action_dim]
            
            # 准备输入
            images, img_masks = self._preprocess_images(batch)
            tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

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
            
            # 5. 计算特征距离 loss
            feature_loss = torch.mean(F.relu(torch.square(gen_feature - pred_feature) - 0.1))

            # 6. 计算 generated_actions 与 pred_action 在前 attn_act_len 步的 L2 距离
            # 对齐形状到 [batch, attn_act_len, original_action_dim]
            pred_actions_slice = pred_action[:, :attn_act_len, :original_action_dim]
            # 每一步在 action 维度上的 L2 范数，然后对 batch 和时间步求平均
            l2_per_step = torch.linalg.vector_norm(gen_actions_slice - pred_actions_slice, dim=-1)  # [batch, attn_act_len]
            l2_actions = l2_per_step.mean()

            loss_dict = {
                "feature_loss": feature_loss.item(),
                "l2_actions": l2_actions.item(),
            }
            
            return feature_loss, loss_dict

        else:
            if cmp:
                self.freeze_params()
            if not cmp:
                self.unfreeze_params()

            # Prepare inputs
            images, img_masks = self._preprocess_images(batch)
            tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

            actions = self.prepare_action(batch)
            if not cmp:
                # Compute loss (no separate state needed for PI05)
                losses = self.model.forward(images, img_masks, tokens, masks, actions)
                losses = losses[:, :, :original_action_dim]
                loss = losses.mean()
                loss_dict = {
                    "loss": loss.item(),
                }
            else:
                losses = self.model.forward_cmp(images, img_masks, tokens, masks)
                loss = losses.mean()
                loss_dict = {
                    "cmp_loss": loss.item(),
                }



            return loss, loss_dict

    def get_action_context(self, batch_size: int = 1, prev_steps: int | None = None, pred_steps: int | None = None) -> dict:
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
        # 从 config 读取默认值
        if prev_steps is None:
            prev_steps = self.config.attn_act_len
        if pred_steps is None:
            pred_steps = self.config.attn_act_len

        # 获取 action_dim 和 device
        original_action_dim = self.config.output_features[ACTION].shape[0]
        device = next(self.parameters()).device

        chunk_size = self.config.chunk_size
        
        # 初始化为零填充的数据（始终返回正确形状，不返回 None）
        prev_actions = torch.zeros(batch_size, prev_steps, original_action_dim, device=device, dtype=torch.float32)
        pred_action = torch.zeros(batch_size, pred_steps, original_action_dim, device=device, dtype=torch.float32)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 如果有预测 buffer，填充真实数据
        if self.step_num_in_chunk is not None:

            valid_mask = (self.step_num_in_chunk > prev_steps) & (self.step_num_in_chunk + pred_steps <= chunk_size)
            
            # 为有效的样本填充真实数据
            for i in range(batch_size):
                if valid_mask[i]:
                    prev_actions[i] = self._last_actions_list[i, :, :original_action_dim]
                    pred_action[i] = self._predicted_actions_buffer[i, :pred_steps, :original_action_dim]
        
        result = {
            'prev_actions': prev_actions,
            'pred_action': pred_action,
            'actions_seq_valid': valid_mask,
        }

        return result
