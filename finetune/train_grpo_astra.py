# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import argparse
import json
import math
import os
import time
from typing import Optional
from collections import deque
from contextlib import nullcontext
import pdb
from datetime import datetime

import cv2
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import wandb
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from fastvideo.utils.fsdp_util import apply_fsdp_checkpointing
from fastvideo.utils.logging_ import main_print
from fastvideo.utils.parallel_states import (
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
)
from PIL import Image
import imageio.v2 as imageio
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from diffsynth.models import ModelManager
from diffsynth.pipelines import WanVideoAstraPipeline
from diffsynth.models.wan_video_dit_moe import WanModelMoe, DiTBlockWithMoE
from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.configs.model_config import model_loader_configs

from fastvideo.dataset.spatialvid_datasets import SpatialVidFramePackDataset, framepack_collate_fn
from reward_model.camera_alignment_reward import CameraAlignmentReward

from get_target_pose_qwj import extract_target_poses_to_txt

def parse_path_list(path_value):
    if path_value is None:
        return []
    return [item.strip() for item in str(path_value).split(",") if item.strip()]


def resolve_model_files(path_value, default_candidates=None):
    resolved_paths = []
    for entry in parse_path_list(path_value):
        if os.path.isdir(entry):
            candidates = default_candidates or []
            found = False
            for candidate in candidates:
                candidate_path = os.path.join(entry, candidate)
                if os.path.exists(candidate_path):
                    resolved_paths.append(candidate_path)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(
                    f"No matching checkpoint file found in directory: {entry}."
                )
        else:
            if not os.path.exists(entry):
                raise FileNotFoundError(f"Checkpoint file not found: {entry}")
            resolved_paths.append(entry)
    return resolved_paths


def replace_dit_model_in_manager():
    """Replace DiT model class with camera version"""
    from diffsynth.models.wan_video_dit_cam import WanModelCam
    from diffsynth.configs.model_config import model_loader_configs

    for i, config in enumerate(model_loader_configs):
        keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource = config

        if 'wan_video_dit' in model_names:
            new_model_names = []
            new_model_classes = []

            for name, cls in zip(model_names, model_classes):
                if name == 'wan_video_dit':
                    new_model_names.append(name)
                    new_model_classes.append(WanModelCam)
                else:
                    new_model_names.append(name)
                    new_model_classes.append(cls)

            model_loader_configs[i] = (keys_hash, keys_hash_with_shape, new_model_names, new_model_classes, model_resource)

def add_framepack_components(dit_model):
    """添加FramePack相关组件"""
    if not hasattr(dit_model, 'clean_x_embedder'):
        inner_dim = dit_model.blocks[0].self_attn.q.weight.shape[0]
        
        class CleanXEmbedder(nn.Module):
            def __init__(self, inner_dim):
                super().__init__()
                self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
                self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
            
            def forward(self, x, scale="1x"):
                if scale == "1x":
                    x = x.to(self.proj.weight.dtype)
                    return self.proj(x)
                elif scale == "2x":
                    x = x.to(self.proj_2x.weight.dtype)
                    return self.proj_2x(x)
                elif scale == "4x":
                    x = x.to(self.proj_4x.weight.dtype)
                    return self.proj_4x(x)
                else:
                    print(f"❌ Unsupported scale: {scale}")
                    raise ValueError(f"Unsupported scale: {scale}")
        
        dit_model.clean_x_embedder = CleanXEmbedder(inner_dim)
        model_dtype = next(dit_model.parameters()).dtype
        dit_model.clean_x_embedder = dit_model.clean_x_embedder.to(dtype=model_dtype)
        print("✅ 添加了FramePack的clean_x_embedder组件")
        
def add_cpe_components(dit_model, moe_config):
    """🔧 添加MoE相关组件 - 修正版本"""
    if not hasattr(dit_model, 'moe_config'):
        dit_model.moe_config = moe_config
        print("✅ 添加了MoE配置到模型")
    dit_model.top_k = moe_config.get("top_k", 1)

    # 为每个block动态添加MoE组件
    dim = dit_model.blocks[0].self_attn.q.weight.shape[0]
    unified_dim = moe_config.get("unified_dim", 25)
    num_experts = moe_config.get("num_experts", 4)
    from diffsynth.models.wan_video_dit_moe import ModalityProcessor, MultiModalMoE
    dit_model.sekai_processor = ModalityProcessor("sekai", 13, unified_dim)
    dit_model.nuscenes_processor = ModalityProcessor("nuscenes", 8, unified_dim)
    dit_model.openx_processor = ModalityProcessor("openx", 13, unified_dim)  # OpenX使用13维输入，类似sekai但独立处理
    dit_model.global_router = nn.Linear(unified_dim, num_experts)


    for i, block in enumerate(dit_model.blocks):
        # MoE网络 - 输入unified_dim，输出dim
        block.moe = MultiModalMoE(
            unified_dim=unified_dim,
            output_dim=dim,  # 输出维度匹配transformer block的dim
            num_experts=moe_config.get("num_experts", 4),
            top_k=moe_config.get("top_k", 2)
        )

        print(f"✅ Block {i} 添加了MoE组件 (unified_dim: {unified_dim}, experts: {moe_config.get('num_experts', 4)})")

def call_wan_cpe_dit(
    transformer,
    latents,
    timesteps,
    context,
    cam_emb=None,
    model_kwargs=None,
):
    timestep_tensor = timesteps.to(latents.device, dtype=torch.float32)

    forward_kwargs = {
        "timestep": timestep_tensor,
        "context": context,
    }

    if cam_emb is not None:
        forward_kwargs["cam_emb"] = cam_emb

    if model_kwargs:
        for key, value in model_kwargs.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.to(latents.device)
            forward_kwargs[key] = value

    outputs = transformer(latents, **forward_kwargs)

    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs

def to_device(batch, device, dtype=None):
    def _move(x):
        if isinstance(x, torch.Tensor):
            target = x.to(device)
            return target.to(dtype) if dtype and target.dtype != dtype else target
        if isinstance(x, dict):
            return {k: _move(v) for k, v in x.items()}
        return x
    return _move(batch)

def repeat_for_group(obj, repeats):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.repeat_interleave(repeats, dim=0)
    if isinstance(obj, dict):
        return {k: repeat_for_group(v, repeats) for k, v in obj.items()}
    if isinstance(obj, list):
        return [item for item in obj for _ in range(repeats)]
    if isinstance(obj, tuple):
        return tuple(repeat_for_group(list(obj), repeats))
    return obj


def video_first_frame_to_pil(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return None

    ret, frame = cap.read()
    if not ret:
        print("无法读取视频的第一帧")
        cap.release()
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(frame_rgb)

    cap.release()

    return pil_image

def to_tensor(value, device, dtype=None, batch_dim=True):
    """确保值为tensor并转移到指定设备"""
    if value is None:
        return None
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.numel() == 0:
        return None
    
    if dtype is None and tensor.dtype.is_floating_point:
        dtype = None  # 使用原dtype
    
    tensor = tensor.to(device=device, dtype=dtype)
    if batch_dim and tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)
    elif not batch_dim and tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif not batch_dim and tensor.dim() == 0:
        tensor = tensor.view(1, 1)
    
    return tensor


def prepare_condition_for_sample(batch, sample_index, device, model_dtype):
    """
    为单个样本的采样准备条件
    Args:
        batch: 批次数据
        sample_index: 样本索引
        device: 设备
        model_dtype: 模型数据类型
    
    Returns:
        tuple: (conditioning, gt_extrinsics_tensor, prompt_text)
    """
    # 提取数据集信息
    dataset_type = batch.get("dataset_type", ["sekai"])[0]
    dataset_name = batch.get("dataset_name", ["unknown"])[0]
    
    # 准备相机嵌入
    cam_emb = None
    camera_batch = batch.get("camera")
    if camera_batch is not None:
        cam_value = camera_batch[sample_index]
        cam_emb = to_tensor(cam_value, device, model_dtype, batch_dim=False)
        if cam_emb.dim() == 2:
            cam_emb = cam_emb.unsqueeze(0)
    
    # 准备模态输入
    modality_inputs = {}
    if cam_emb is not None:
        modality_key = dataset_type if dataset_type in ("sekai", "nuscenes", "openx") else "sekai"
        modality_inputs[modality_key] = cam_emb.clone()
    modality_inputs = modality_inputs or None
    
    # 准备latents
    latent_keys = [
        ("latents", None), ("clean_latents", None),
        ("clean_latents_2x", None), ("clean_latents_4x", None),
        ("latent_indices", torch.long), ("clean_latent_indices", torch.long),
        ("clean_latent_2x_indices", torch.long), ("clean_latent_4x_indices", torch.long)
    ]
    
    framepack_pairs = {}
    for key, dtype in latent_keys:
        if key in batch:
            value = batch[key][sample_index]
            if "indices" in key:
                framepack_pairs[key] = to_tensor(value, device, dtype, batch_dim=False)
            else:
                framepack_pairs[key] = to_tensor(value, device, model_dtype)
    
    # 准备文本和图像嵌入
    prompt_text = batch["prompt_text"][sample_index]
    context = batch["prompt_emb"]["context"][sample_index].to(device)
    
    image_emb = batch.get("image_emb", {})
    image_tensor = None
    if "clip_feature" in image_emb or "y" in image_emb:
        image_tensor = image_emb.get("clip_feature", image_emb.get("y"))[sample_index].to(device)
    
    # 准备model_kwargs
    model_kwargs = {k: v for k, v in framepack_pairs.items() if v is not None}
    model_kwargs["context"] = context
    if image_tensor is not None:
        model_kwargs["image_emb"] = image_tensor
    
    # 准备ground truth extrinsics
    gt_poses = batch.get("gt_absolute_poses", [None])[sample_index]
    if gt_poses is None or not isinstance(gt_poses, list):
        raise ValueError("Batch must contain 'gt_absolute_poses' as a list of tensors.")
    gt_extrinsics_tensor = torch.stack(gt_poses, dim=0).to(device=device, dtype=torch.float32)
    
    conditioning = {
        "cam_emb": cam_emb,
        "modality_inputs": modality_inputs,
        "model_kwargs": model_kwargs,
    }
    
    return conditioning, gt_extrinsics_tensor, prompt_text

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)
    

def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 


    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = ((
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean,pred_original_sample



def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"

def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        conditioning,
        grpo_sample,
    ):

    if grpo_sample:
        # 准备 WanModelMoe 的输入
        model_dtype = next(transformer.parameters()).dtype

        model_kwargs = conditioning.get("model_kwargs", {})
        context_raw = model_kwargs.get("context")
        if context_raw is None:
            raise ValueError("conditioning['model_kwargs'] must contain 'context'.")
        context = context_raw.to(device=z.device, dtype=model_dtype)

        cam_emb = conditioning.get("cam_emb")
        if isinstance(cam_emb, torch.Tensor):
            cam_emb = cam_emb.to(device=z.device, dtype=model_dtype)

        prepared_model_kwargs = {}
        def _prepare_value(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                target_dtype = model_dtype if value.dtype.is_floating_point else value.dtype
                return value.to(device=z.device, dtype=target_dtype)
            if isinstance(value, dict):
                return {k: _prepare_value(v) for k, v in value.items() if v is not None}
            if isinstance(value, list):
                return [_prepare_value(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_prepare_value(v) for v in value)
            return value

        for key, value in model_kwargs.items():
            if key == "context":
                prepared_model_kwargs[key] = context
            else:
                prepared_model_kwargs[key] = _prepare_value(value)

        forward_model_kwargs = {
            k: v for k, v in prepared_model_kwargs.items() if k != "context"
        }

        # 进行去噪推理步骤，每步去噪被视为一个马尔可夫过程
        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:
            sigma = sigma_schedule[i]
            timestep_value = float(int(sigma * 1000))
            timesteps = torch.full(
                [z.shape[0]],
                timestep_value,
                device=z.device,
                dtype=torch.float32,
            )

            transformer.eval()
            with (
                torch.autocast("cuda", torch.bfloat16)
                if z.device.type == "cuda"
                else nullcontext()
            ):
                pred = call_wan_cpe_dit(
                    transformer,
                    z,
                    timesteps,
                    context,
                    cam_emb=cam_emb,
                    model_kwargs=forward_model_kwargs,
                ).to(torch.float32)

            z, pred_original, log_prob = flux_step(
                pred.to(torch.float32),
                z.to(torch.float32),
                args.eta,
                sigmas=sigma_schedule,
                index=i,
                prev_sample=None,
                grpo=True,
                sde_solver=True,
            )
            z = z.to(dtype=model_dtype)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1)
        return z, latents, all_latents, all_log_probs

        
def grpo_one_step(
            args,
            latents,
            pre_latents,
            transformer,
            timesteps,
            step_index,
            sigma_schedule,
            conditioning,
):
    transformer.train()
    model_dtype = next(transformer.parameters()).dtype

    model_kwargs = conditioning.get("model_kwargs", {})
    context_raw = model_kwargs.get("context")
    if context_raw is None:
        raise ValueError("conditioning['model_kwargs'] must contain 'context'.")
    context = context_raw.to(device=latents.device, dtype=model_dtype)

    cam_emb = conditioning.get("cam_emb")
    if isinstance(cam_emb, torch.Tensor):
        cam_emb = cam_emb.to(device=latents.device, dtype=model_dtype)

    prepared_model_kwargs = {}
    def _prepare_value(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            target_dtype = model_dtype if value.dtype.is_floating_point else value.dtype
            return value.to(device=latents.device, dtype=target_dtype)
        if isinstance(value, dict):
            return {k: _prepare_value(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [_prepare_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_prepare_value(v) for v in value)
        return value

    for key, value in model_kwargs.items():
        if key == "context":
            prepared_model_kwargs[key] = context
        else:
            prepared_model_kwargs[key] = _prepare_value(value)

    forward_model_kwargs = {
        k: v for k, v in prepared_model_kwargs.items() if k != "context"
    }

    with (
        torch.autocast("cuda", torch.bfloat16)
        if latents.device.type == "cuda"
        else nullcontext()
    ):
        pred = call_wan_cpe_dit(
            transformer,
            latents,
            timesteps.to(torch.float32),
            context,
            cam_emb=cam_emb,
            model_kwargs=forward_model_kwargs,
        ).to(torch.float32)


    z, pred_original, log_prob = flux_step(
        pred.to(torch.float32),
        latents.to(torch.float32),
        args.eta,
        sigma_schedule,
        step_index,
        prev_sample=pre_latents.to(torch.float32),
        grpo=True,
        sde_solver=True,
    )
    return log_prob



def sample_reference_model(
    args,
    device, 
    pipe,
    batch, 
    reward_model,
    tokenizer,
    preprocess_val,
    train_step
):
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)

    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    _ = (tokenizer, preprocess_val)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    latents_batch = batch["latents"] # [train_batch_size * num_generatios, C, T, H, W]（use group）
    if isinstance(latents_batch, torch.Tensor):
        B = latents_batch.shape[0]
    else:
        B = len(latents_batch)

    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 4
    IN_CHANNELS = 16
    latent_t = (t - 1) // TEMPORAL_DOWNSAMPLE
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), max(B // batch_size, 1))

    all_latents = []
    all_log_probs = []
    all_rewards = []  
    model_dtype = next(pipe.dit.parameters()).dtype
    conditioning_records = []

    # 为每个样本初始化相同噪声
    if args.init_same_noise:
        input_latents = torch.randn(
                (1, IN_CHANNELS, latent_t, latent_h, latent_w),  # (c,t,h,w)
                device=device,
                dtype=model_dtype,
            )

    for index, batch_idx in enumerate(batch_indices):
        sample_index = batch_idx[0].item()
        conditioning, gt_extrinsics, prompt_text = prepare_condition_for_sample(
            batch, sample_index, device, model_dtype
        )
        conditioning_records.append({
            "conditioning": conditioning,
            "gt_extrinsics": gt_extrinsics,
        })
        
        # 为每个样本的初始噪声加入微小扰动
        if args.init_same_noise:
            input_latents = input_latents + torch.randn_like(input_latents) * 0.02
        
        # 为每个样本初始化不同噪声
        if not args.init_same_noise:
            input_latents = torch.randn(
                    (1, IN_CHANNELS, latent_t, latent_h, latent_w),  # (c,t,h,w)
                    device=device,
                    dtype=model_dtype,
                )
        
        grpo_sample=True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs = run_sample_step(
                args,
                input_latents.clone(),
                progress_bar,
                sigma_schedule,
                pipe.dit,
                conditioning_records[-1]["conditioning"],
                grpo_sample,
            )
            
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)

        rank = int(os.environ["RANK"])

        video_output_path = f"{args.experiment_dir}/videos/wan_2_1_step_{train_step+1}_rank_{rank}_{index}.mp4"
        conditioning_records[-1]["video_path"] = video_output_path
        
        with torch.inference_mode():
            latents_to_decode = latents.to(dtype=torch.float32)
            decoded_video = pipe.decode_video(latents_to_decode, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
            video_np = decoded_video[0].to(torch.float32).permute(1, 2, 3, 0).cpu().numpy()
            video_np = (video_np * 0.5 + 0.5).clip(0, 1)
            video_np = (video_np * 255).astype(np.uint8)
        try:
            with imageio.get_writer(video_output_path, fps=24) as writer:
                for frame in video_np:
                    writer.append_data(frame)
            print(f"\n 已导出视频到: {video_output_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"在导出视频时发生错误；{e}")
            
        target_pose_output_path = f"{args.experiment_dir}/target_poses/wan_2_1_step_{train_step+1}_rank_{rank}_{index}.txt"
        # extract_target_poses_to_txt(conditioning["cam_emb"].squeeze(0), target_pose_output_path)
        prompt_output_path = f"{args.experiment_dir}/prompts/wan_2_1_step_{train_step+1}_rank_{rank}_{index}.txt"
        with open(prompt_output_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)

        # Calculate 3D-Aware Reward
        gt_extrinsics_tensor = conditioning_records[-1].get("gt_extrinsics")
        if gt_extrinsics_tensor is None:
            raise RuntimeError("Ground-truth extrinsics are required to compute the reward.")
        gt_extrinsics_np = gt_extrinsics_tensor.detach().cpu().numpy()
        reward_info = {}
        try:
            decoded_video = decoded_video.transpose(1, 2)
            video_min = decoded_video.min()
            video_max = decoded_video.max()
            normalized_video = 2 * (decoded_video - video_min) / (video_max - video_min + 1e-8) - 1
            reward_info = reward_model.calculate_reward(normalized_video, gt_extrinsics_np)
            
            # print("\n--- 对齐奖励结果 ---")
            # print(f"平均旋转误差: {reward_info['mean_rotation_error_degrees']:.2f} 度")
            # print(f"平均平移误差 (尺度对齐后): {reward_info['mean_translation_error']:.4f}")
            # print(f"计算出的轨迹尺度因子: {reward_info['translation_scale_factor']:.4f}")
            # print("-" * 20)
            # print(f"旋转奖励: {reward_info['rotation_reward']:.4f}")
            # print(f"平移奖励: {reward_info['translation_reward']:.4f}")
            # print(f"最终加权总奖励: {reward_info['total_reward']:.4f}")
            # print("----------------------\n")
            
            with open(target_pose_output_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- 对齐奖励结果 ---\n")
                f.write(f"平均旋转误差: {reward_info['mean_rotation_error_degrees']:.2f} 度\n")
                f.write(f"平均平移误差 (尺度对齐后): {reward_info['mean_translation_error']:.4f}\n")
                f.write(f"计算出的轨迹尺度因子: {reward_info['translation_scale_factor']:.4f}\n")
                f.write("-" * 20 + "\n")
                f.write(f"旋转奖励: {reward_info['rotation_reward']:.4f}\n")
                f.write(f"平移奖励: {reward_info['translation_reward']:.4f}\n")
                f.write(f"最终加权总奖励: {reward_info['total_reward']:.4f}\n")
                f.write("----------------------\n")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"在计算奖励时发生错误: {e}")

        if not reward_info:
            raise ValueError("reward_info calculation fault")

        if isinstance(reward_info, dict):
            total_reward_value = float(reward_info.get("total_reward", 0.0))
        else:
            total_reward_value = float(reward_info)
            reward_info = {"total_reward": total_reward_value}

        reward_tensor = torch.tensor([total_reward_value], device=device, dtype=torch.float32)
        all_rewards.append(reward_tensor)
        conditioning_records[-1]["reward_info"] = reward_info

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    if not all_rewards:
        raise RuntimeError(
            "No rewards were computed during sampling. Ensure CameraAlignmentReward returns a total_reward value."
        )
    all_rewards = torch.cat(all_rewards, dim=0)

    return all_rewards, all_latents, all_log_probs, sigma_schedule, conditioning_records


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def train_one_step(
    args,
    device,
    pipe,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    batch,
    noise_scheduler,
    max_grad_norm,
    preprocess_val,
    train_step,
):

    total_loss = 0.0
    optimizer.zero_grad()
    model_dtype = next(pipe.dit.parameters()).dtype
    batch = to_device(batch, device, model_dtype)

    if args.use_group:
        # train_batch_size -> train_batch_size * num_generations
        batch = repeat_for_group(batch, args.num_generations)

    reward, all_latents, all_log_probs, sigma_schedule, conditioning_records = sample_reference_model(
            args,
            device,
            pipe,
            batch,
            reward_model,
            tokenizer,
            preprocess_val,
            train_step=train_step,
        )
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps =  torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[
            :, :-1
        ][:, :-1],  # each entry is the latent before timestep t
        "next_latents": all_latents[
            :, 1:
        ][:, :-1],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        "rewards": reward.to(torch.float32),
    }
    gathered_reward = gather_tensor(samples["rewards"])
    if dist.get_rank()==0:
        print("gathered_reward", gathered_reward)
        with open(f'{args.experiment_dir}/reward.txt', 'a') as f: 
            f.write(f"{gathered_reward.mean().item()}\n")

    #计算advantage
    if args.use_group:
        n = len(samples["rewards"]) // (args.num_generations)
        advantages = torch.zeros_like(samples["rewards"])
        
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        
        samples["advantages"] = advantages
    else:
        advantages = (samples["rewards"] - gathered_reward.mean())/(gathered_reward.std()+1e-8)
        samples["advantages"] = advantages

    
    perms = torch.stack(
        [
            torch.randperm(len(samples["timesteps"][0]))
            for _ in range(batch_size)
        ]
    ).to(device) 
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device) [:, None],
            perms,
        ]
    samples_batched = {
        k: v.unsqueeze(1)
        for k, v in samples.items()
    }
    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]
    train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)
    for i, sample in list(enumerate(samples_batched_list)):
        conditioning = conditioning_records[i]["conditioning"]
        for step_idx in range(train_timesteps):
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            new_log_probs = grpo_one_step(
                args,
                sample["latents"][:, step_idx],
                sample["next_latents"][:, step_idx],
                pipe.dit,
                sample["timesteps"][:, step_idx],
                perms[i][step_idx],
                sigma_schedule,
                conditioning,
            )

            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            ratio = torch.exp(new_log_probs - sample["log_probs"][:, step_idx])

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * train_timesteps)

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
        
        if (i+1)%args.gradient_accumulation_steps==0:
            grad_norm = torch.nn.utils.clip_grad_norm_(pipe.dit.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if dist.get_rank()%8==0:
            print(f"第 {i} 个动作对目标函数的贡献:")
            print("reward", sample["rewards"].item())
            print("ratio", ratio)
            print("advantage", sample["advantages"].item())
            print("final loss", loss.item())
        dist.barrier()
    
    return total_loss, grad_norm.item()


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)
    else:
        raise ValueError(f"local_rank {local_rank} 超出可用GPU范围")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"World Size: {world_size}, Rank: {rank}, Local Rank: {local_rank}")
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    initialize_sequence_parallel_state(args.sp_size)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"/mnt/data/louis_crq/DanceGRPO/dancegrpo_experiment_qwj_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    sub_dirs = ["videos", "target_poses", "prompts","checkpoints"]
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(experiment_dir, sub_dir), exist_ok=True)    
    args.experiment_dir = experiment_dir

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    preprocess_val = None
    processor = None
    
    main_print("--> Loading CameraAlignmentReward Model")
    reward_model = CameraAlignmentReward(rank=rank, device=device)

    replace_dit_model_in_manager()
    
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "/mnt/data/louis_crq/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "/mnt/data/louis_crq/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "/mnt/data/louis_crq/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoAstraPipeline.from_model_manager(model_manager, device="cuda")

    add_framepack_components(pipe.dit)
    
    from diffsynth.models.wan_video_dit_cam import Cam_Encoder, Cam_Processor
    pipe.dit.cam_processor = Cam_Processor(13, 25) # project the input dim to unified dim
    
    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for i, block in enumerate(pipe.dit.blocks):
        # Add Camera Pose Encoder
        block.cpe = Cam_Encoder(
            unified_dim=25,
            output_dim=dim
        )
        block.projector = nn.Linear(dim, dim)
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))

    key_parts_map = {
        'sekai_processor': 'cam_processor',
        'moe.experts.0': 'cpe.encoder'
    }
    dit_state_dict = torch.load(args.our_checkpoint_path, map_location="cpu")
    
    # modify parameters name
    keys_to_replace = []
    for old_key in dit_state_dict.keys():
        for old_key_parts in key_parts_map.keys():
            if isinstance(old_key, str) and old_key_parts in old_key:
                keys_to_replace.append(old_key)
                
    for old_key in keys_to_replace:
        value = dit_state_dict[old_key]
        for old_key_parts, new_key_parts in key_parts_map.items():
            if old_key_parts in old_key:
                new_key = old_key.replace(old_key_parts, new_key_parts)
                del dit_state_dict[old_key]
                dit_state_dict[new_key] = value
    
    pipe.dit.load_state_dict(dit_state_dict, strict=False)  # 使用strict=False以兼容新增的MoE组件
    
    pipe.dit.requires_grad_(False)
    for name, module in pipe.dit.named_modules():
        if any(keyword in name for keyword in ["cam_encoder", "projector", "clean_x_embedder", "cpe", "cam_processor"]):
            for param in module.parameters():
                param.requires_grad = True
    
    pipe = pipe.to(device)
    model_dtype = next(pipe.dit.parameters()).dtype
    
    transformer = pipe.dit

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, (DiTBlockWithMoE,), args.selective_checkpointing
        )

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    # Load the reference model
    main_print(f"--> model loaded")

    # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
    
    trainable_params = []
    total_params = 0

    for name, param in transformer.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params.append((name, param.numel()))

    print("Trainable parameters by module:")
    # for name, count in trainable_params:
    #     if "blocks.0" in name:
    #         print(f"{name:<60} {count:,}")

    num_trainable = sum(c for _, c in trainable_params)
    print(f"\nTotal trainable parameters: {num_trainable:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable ratio: {100 * num_trainable / total_params:.2f}%")

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )
    
    train_dataset = SpatialVidFramePackDataset(
        args.dataset_path,
        video_info_path="/mnt/data/louis_crq/data/preprocess_data/SpatialVID-HQ/manifest25.json",
        steps_per_epoch=args.steps_per_epoch,
        min_condition_frames=args.min_condition_frames,
        max_condition_frames=args.max_condition_frames,
        target_frames=args.target_frames,
    )
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
    )


    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=framepack_collate_fn,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    
    #vae.enable_tiling()

    if rank <= 0:
        project = "wan_2_1"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )


    step_times = deque(maxlen=100)

    # The number of epochs 1 is a random value; you can also set the number of epochs to be two.
    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch       
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            if (step-1) % args.checkpointing_steps == 0 and step != 1:
                cpu_state = transformer.state_dict()
                if rank <= 0:
                    save_dir = os.path.join(f"{args.experiment_dir}/checkpoints", f"checkpoint-{step}-{epoch}")
                    os.makedirs(save_dir, exist_ok=True)
                    weight_path = os.path.join(save_dir,
                                            "diffusion_pytorch_model.ckpt")
                    torch.save(cpu_state, weight_path)
                    main_print(f"--> checkpoint saved at step {step}: {save_dir}")
                dist.barrier()
            if step > args.max_train_steps:
                break
            loss, grad_norm = train_one_step(
                args,
                device, 
                pipe,
                reward_model,
                processor,
                optimizer,
                lr_scheduler,
                batch,
                noise_scheduler,
                args.max_grad_norm,
                preprocess_val,
                train_step=step
            )

    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                    },
                    step=step,
                )

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/data/louis_crq/DanceGRPO/data/outputs/grpo",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether compute advantages for each prompt",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep downsample ratio",
    )
    parser.add_argument(
        "--clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    )
    parser.add_argument(
        "--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    )
    parser.add_argument(
        "--cfg_infer",
        type = float,
        default=5.0,
        help="cfg for training",
    )
    
    # Train our model
    parser.add_argument(
        "--our_checkpoint_path",
        type=str,
        default="checkpoints/our_model.ckpt",
        help="Path to save our model checkpoints.",
    )
    parser.add_argument(
        "--moe_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for MoE.",
    )
    parser.add_argument(
        "--steps_per_epoch", 
        type=int, 
        default=200000
    )
    parser.add_argument(
        "--max_epochs", 
        type=int, 
        default=100000
    )
    parser.add_argument(
        "--min_condition_frames", 
        type=int, 
        default=8, 
        help="最小条件帧数"
    )
    parser.add_argument(
        "--max_condition_frames", 
        type=int, 
        default=120, 
        help="最大条件帧数"
    )
    parser.add_argument(
        "--target_frames", 
        type=int, 
        default=32, 
        help="目标帧数"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/share_zhuyixuan05/zhuyixuan05/spatialvid",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank for LoRA adapters.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="Alpha for LoRA adapters.",
    )

    args = parser.parse_args()
    main(args)
