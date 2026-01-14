import torch
import torch.nn as nn
import lightning as pl
import wandb
import os
import copy
from diffsynth import WanVideoAstraPipeline, ModelManager
import json
import numpy as np
from PIL import Image
import imageio
import random
from torchvision.transforms import v2
from einops import rearrange
from pose_classifier import PoseClassifier
from scipy.spatial.transform import Rotation as R
import traceback
import argparse
from utils.dataset_utils import generate_random_camera_poses

def quat_to_rot_matrix(quat):
    """
    Convert quaternion to 3x3 rotation matrix (unit quaternion, format: [qx, qy, qz, qw])
    """
    qx, qy, qz, qw = quat.unbind(dim=-1)
    # Calculate rotation matrix elements
    R00 = 1 - 2 * (qy**2 + qz**2)
    R01 = 2 * (qx*qy - qz*qw)
    R02 = 2 * (qx*qz + qy*qw)
    
    R10 = 2 * (qx*qy + qz*qw)
    R11 = 1 - 2 * (qx**2 + qz**2)
    R12 = 2 * (qy*qz - qx*qw)
    
    R20 = 2 * (qx*qz - qy*qw)
    R21 = 2 * (qy*qz + qx*qw)
    R22 = 1 - 2 * (qx**2 + qy**2)
    
    # Combine into 3x3 matrix (shape: [..., 3, 3])
    rot_matrix = torch.stack([
        torch.stack([R00, R01, R02], dim=-1),
        torch.stack([R10, R11, R12], dim=-1),
        torch.stack([R20, R21, R22], dim=-1)
    ], dim=-2)
    return rot_matrix

def vec7_to_4x4(pose_vec):
    """
    Convert 7-element vector [tx, ty, tz, qx, qy, qz, qw] to 4x4 camera extrinsic matrix
    """
    # Separate translation vector and quaternion (input shape: [7] or [N, 7])
    trans = pose_vec[..., :3]  # Translation: [tx, ty, tz]
    quat = pose_vec[..., 3:7]  # Quaternion: [qx, qy, qz, qw]
    
    # Quaternion to 3x3 rotation matrix
    rot_matrix = quat_to_rot_matrix(quat)  # Shape: [..., 3, 3]
    
    # Expand translation vector to [3, 1] shape (concatenate with rotation matrix to form 3x4)
    trans = trans.unsqueeze(-1)  # Shape: [..., 3, 1]
    rot_trans = torch.cat([rot_matrix, trans], dim=-1)  # Shape: [..., 3, 4]
    
    # Create 4th row [0, 0, 0, 1], adapting to batch dimensions
    # Generate batch dimensions matching input (e.g., if input is [N,7], then 4th row shape is [N,1,4])
    batch_dims = rot_trans.shape[:-2]  # Get batch dimensions (e.g., [N])
    fourth_row = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=rot_trans.device)
    # Expand 4th row to [batch_dims..., 1, 4]
    fourth_row = fourth_row.view(*((1,)*len(batch_dims)), 1, 4).repeat(*batch_dims, 1, 1)
    
    # Concatenate 3x4 matrix and 4th row to get 4x4 matrix
    transform_matrix = torch.cat([rot_trans, fourth_row], dim=-2)  # Shape: [..., 4, 4]
    return transform_matrix

def compute_absolute_pose_matrixes(poses):
    """
    Compute absolute pose matrices for a series of camera poses, returning a list where each element is a (4, 4) tensor
    
    Parameters:
    poses: Array of shape (N, 7), each row represents a camera pose [tx, ty, tz, qx, qy, qz, qw]
    
    Returns:
    absolute_matrices: List where each element is a (4, 4) PyTorch tensor representing a homogeneous transformation matrix
    """
    # Convert input to numpy array (maintaining original processing logic)
    absolute_matrices = []
    
    for pose in poses:
        # Separate translation vector and quaternion
        t = pose[:3]  # Translation [tx, ty, tz]
        q = pose[3:]  # Quaternion [qx, qy, qz, qw]
        
        # Compute rotation matrix R
        rot = R.from_quat(q)
        R_mat = rot.as_matrix()  # 3×3 rotation matrix
        
        # Combine into 4×4 homogeneous transformation matrix (numpy array)
        mat_3x4 = np.hstack([R_mat, t.reshape(3, 1)])
        homogeneous_mat_np = np.vstack([mat_3x4, np.array([0, 0, 0, 1])])
        
        # Convert numpy matrix to PyTorch tensor and add to list
        homogeneous_mat_tensor = torch.from_numpy(homogeneous_mat_np).to(torch.float64)
        absolute_matrices.append(homogeneous_mat_tensor)
    
    return absolute_matrices  # List with (4,4) tensor elements

def compute_relative_pose_matrix(pose1, pose2):
    """
    Compute relative pose between two adjacent frames, returning a 3×4 camera matrix [R_rel | t_rel]
    
    Parameters:
    pose1: Camera pose of frame i, shape (7,) array [tx1, ty1, tz1, qx1, qy1, qz1, qw1]
    pose2: Camera pose of frame i+1, shape (7,) array [tx2, ty2, tz2, qx2, qy2, qz2, qw2]
    
    Returns:
    relative_matrix: 3×4 relative pose matrix, first 3 columns are rotation matrix R_rel, 4th column is translation vector t_rel
    """
    pose1 = pose1.detach().to(torch.float64).cpu().numpy()
    pose2 = pose2.detach().to(torch.float64).cpu().numpy()
    
    # Separate translation vectors and quaternions
    t1 = pose1[:3]  # Frame i translation [tx1, ty1, tz1]
    q1 = pose1[3:]  # Frame i quaternion [qx1, qy1, qz1, qw1]
    t2 = pose2[:3]  # Frame i+1 translation
    q2 = pose2[3:]  # Frame i+1 quaternion
    
    # 1. Compute relative rotation matrix R_rel
    rot1 = R.from_quat(q1)  # Frame i rotation
    rot2 = R.from_quat(q2)  # Frame i+1 rotation
    rot_rel = rot2 * rot1.inv()  # Relative rotation = next frame rotation × inverse of previous frame rotation
    R_rel = rot_rel.as_matrix()  # Convert to 3×3 matrix
    
    # 2. Compute relative translation vector t_rel
    R1_T = rot1.as_matrix().T  # Transpose of previous frame rotation matrix (equivalent to inverse)
    t_rel = R1_T @ (t2 - t1)   # Relative translation = R1^T × (t2 - t1)
    
    # 3. Combine into 3×4 matrix [R_rel | t_rel]
    relative_matrix = np.hstack([R_rel, t_rel.reshape(3, 1)])
    
    return relative_matrix

def framepack_collate_fn(batch):
    def collate_value(values):
        first = values[0]
        if isinstance(first, torch.Tensor):
            if all(isinstance(v, torch.Tensor) and v.shape == first.shape for v in values):
                return torch.stack(values, dim=0)
            return values  # 形状不固定的张量保留列表
        if isinstance(first, dict):
            return {k: collate_value([v[k] for v in values]) for k in first.keys()}
        if isinstance(first, (int, float)):
            return torch.tensor(values)
        return values  # 字符串等保持列表

    return {key: collate_value([sample[key] for sample in batch]) for key in batch[0]}

class SpatialVidFramePackDataset(torch.utils.data.Dataset):
    """SpatialVid dataset supporting FramePack mechanism"""
    
    def __init__(self, base_path, video_info_path, steps_per_epoch, 
                 min_condition_frames=10, max_condition_frames=40,
                 target_frames=10, height=900, width=1600):
        self.base_path = base_path
        self.video_info_path = video_info_path
        self.scenes_path = base_path
        self.min_condition_frames = min_condition_frames
        self.max_condition_frames = max_condition_frames
        self.target_frames = target_frames
        self.height = height
        self.width = width
        self.steps_per_epoch = steps_per_epoch
        self.pose_classifier = PoseClassifier()
        self.scene2prompt = {}
        
        # VAE time compression ratio
        self.time_compression_ratio = 4  # VAE compresses temporal dimension by 4x
        
        # search for all preprocessed scenarios
        self.scene_dirs = []
        # if os.path.exists(self.scenes_path):
        #     for item in os.listdir(self.scenes_path):
        #         scene_dir = os.path.join(self.scenes_path, item)
        #         if os.path.isdir(scene_dir):
        #             encoded_path = os.path.join(scene_dir, "encoded_video.pth")
        #             if os.path.exists(encoded_path):
        #                 self.scene_dirs.append(scene_dir)
        
        with open(base_path, "r") as f:
            data = json.load(f)
        # extract all 'pth' values
        
        # pth_list = [entry["pth"] for entry in data["entries"]]
        # pth_list = [d["pth"] for d in data["entries"]]
        pth_list = []
        pth_to_videopath_dict = {}
        for d in data["entries"]:
            pth_list.append(d["pth"])
            pth_to_videopath_dict[d["pth"]] = d["video_path"]
        
        print(f"  📁 Found {len(pth_list)} paths in manifest")
        
        with open(video_info_path, "r") as f:
            video_infos = json.load(f)
            
        print(type(video_infos))
            
        video_path_list = []
        videopath_to_prompt_dict = {}
        for video_info in video_infos:
            video_path = video_info.get("video_path", "")
            video_path = os.path.join("/mnt/data/louis_crq/data/SpatialVID-HQ", video_path)
            prompt = video_info.get("prompt", "")
            video_path_list.append(video_path)
            videopath_to_prompt_dict[video_path] = prompt
            
        print(f"  📁 Found {len(video_path_list)} video paths in manifest25")
        
        for pth in pth_list:
            scene_dir = os.path.join("/mnt/data/louis_crq/preprocessed_data/SpatialVID_Wan2", pth)
            if not os.path.exists(scene_dir):
                print(f"  ❌ Path does not exist: {scene_dir}")
                continue
            else:
                self.scene_dirs.append(scene_dir)
            
            # Extract prompt
            video_path = pth_to_videopath_dict.get(pth, None)
            if video_path and video_path in video_path_list:
                prompt = videopath_to_prompt_dict.get(video_path, "")
                self.scene2prompt[scene_dir] = prompt
            elif video_path is None:
                print(f"  ❌ Video path not found for pth: {pth}")
                continue
            elif video_path not in video_path_list:
                print(f"  ❌ Video path for pth not found in video_info: {video_path}")
                continue


        print(f"🔧 Found {len(self.scene_dirs)} SpatialVid scenes")
        assert len(self.scene_dirs) > 0, "No encoded scenes found!"
        assert len(self.scene2prompt) == len(self.scene_dirs), "Prompts and scenes count mismatch!"

    def select_dynamic_segment_framepack(self, full_latents):
        """🔧 FramePack-style dynamic selection of condition and target frames - SpatialVid version"""
        total_lens = full_latents.shape[1]
        
        min_condition_compressed = self.min_condition_frames // self.time_compression_ratio
        max_condition_compressed = self.max_condition_frames // self.time_compression_ratio
        target_frames_compressed = self.target_frames // self.time_compression_ratio
        max_condition_compressed = min(max_condition_compressed, total_lens - target_frames_compressed-1)
        
        ratio = random.random()
        #print('ratio:', ratio)
        if ratio < 0.4:
            condition_frames_compressed = 1
        elif ratio < 0.9:
            condition_frames_compressed = random.randint(min_condition_compressed, max_condition_compressed)
        else:
            condition_frames_compressed = target_frames_compressed
        
        # Ensure sufficient frames
        min_required_frames = condition_frames_compressed + target_frames_compressed
        if total_lens < min_required_frames:
            print(f"Insufficient frames after compression: {total_lens} < {min_required_frames}")
            return None
        
        # Randomly select starting position (based on compressed frame count)
        max_start = total_lens - min_required_frames - 1
        start_frame_compressed = random.randint(0, max_start)
        
        condition_end_compressed = start_frame_compressed + condition_frames_compressed
        target_end_compressed = condition_end_compressed + target_frames_compressed

        # 🔧 FramePack-style index handling
        latent_indices = torch.arange(condition_end_compressed, target_end_compressed)  # Only predict future frames
        
        # 🔧 Generate indices based on actual condition_frames_compressed
        # 1x frames: start frame + last 1 frame
        clean_latent_indices_start = torch.tensor([start_frame_compressed])
        clean_latent_1x_indices = torch.tensor([condition_end_compressed - 1])
        clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices])
        
        # 🔧 2x frames: determined by actual condition length
        if condition_frames_compressed >= 2:
            # Take last 2 frames (if available)
            clean_latent_2x_start = max(start_frame_compressed, condition_end_compressed - 2)
            clean_latent_2x_indices = torch.arange(clean_latent_2x_start-1, condition_end_compressed-1)
        else:
            # If condition frames less than 2, create empty indices
            clean_latent_2x_indices = torch.tensor([], dtype=torch.long)
        
        # 🔧 4x frames: determined by actual condition length, max 16 frames
        if condition_frames_compressed >= 4:
            # Take at most 16 history frames (if available)
            clean_4x_start = max(start_frame_compressed, condition_end_compressed - 16)
            clean_latent_4x_indices = torch.arange(clean_4x_start-3, condition_end_compressed-3)
        else:
            clean_latent_4x_indices = torch.tensor([], dtype=torch.long)
        
        # Corresponding original keyframe indices - SpatialVid specific: every 1 frame instead of 4
        keyframe_original_idx = []
        for compressed_idx in range(start_frame_compressed, target_end_compressed):
            keyframe_original_idx.append(compressed_idx)  # SpatialVid uses 1x interval
        
        return {
            'start_frame': start_frame_compressed,
            'condition_frames': condition_frames_compressed,
            'target_frames': target_frames_compressed,
            'condition_range': (start_frame_compressed, condition_end_compressed),
            'target_range': (condition_end_compressed, target_end_compressed),
            
            # FramePack style indices
            'latent_indices': latent_indices,
            'clean_latent_indices': clean_latent_indices,
            'clean_latent_2x_indices': clean_latent_2x_indices,
            'clean_latent_4x_indices': clean_latent_4x_indices,
            
            'keyframe_original_idx': keyframe_original_idx,
            'original_condition_frames': condition_frames_compressed * self.time_compression_ratio,
            'original_target_frames': target_frames_compressed * self.time_compression_ratio,
        }

    def create_pose_embeddings(self, cam_data, segment_info):
        """🔧 Create SpatialVid-style pose embeddings - camera interval is 1 frame instead of 4"""
        cam_data_seq = cam_data['extrinsic']   # N * 7
        
        # 🔧 Compute camera embeddings for all frames (condition + target)
        # SpatialVid specific: every 1 frame instead of 4
        keyframe_original_idx = segment_info['keyframe_original_idx']
        
        relative_cams = []
        gt_absolute_poses = []

        for idx in keyframe_original_idx:
            if gt_absolute_poses is None:
                if idx < len(cam_data_seq):
                    matrix = vec7_to_4x4(cam_data_seq[idx])
                    gt_absolute_poses.append(torch.as_tensor(matrix))
                else:
                    gt_absolute_poses.append(torch.eye(4)[:3, :])
                
            if idx + 1 < len(cam_data_seq):
                cam_prev = cam_data_seq[idx]
                cam_next = cam_data_seq[idx + 1]  # SpatialVid: every 1 frame
                relative_cam = compute_relative_pose_matrix(cam_prev, cam_next)
                relative_cams.append(torch.as_tensor(relative_cam[:3, :]))
                matrix = vec7_to_4x4(cam_next)
                gt_absolute_poses.append(torch.as_tensor(matrix))
            else:
                # If no next frame, use zero motion
                identity_cam = torch.zeros(3, 4)
                relative_cams.append(identity_cam)
                gt_absolute_poses.append(gt_absolute_poses[:-1])  # Repeat last absolute pose
        
        if len(relative_cams) == 0:
            return None
            
        pose_embedding = torch.stack(relative_cams, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        pose_embedding = pose_embedding.to(torch.bfloat16)

        return pose_embedding, gt_absolute_poses
    
    def create_random_pose_embeddings(self, cam_data, segment_info):
        cam_data_seq = cam_data['extrinsic']   # N * 7
        n = segment_info["condition_frames"]
        m = segment_info['target_frames']
        
        # 🔧 Compute camera embeddings for all frames (condition + target)
        keyframe_original_idx = segment_info['keyframe_original_idx']
        
        relative_cams = []
        gt_absolute_poses = []
        
        for idx in keyframe_original_idx:
            # Use poses from dataset for condition frames
            if not gt_absolute_poses:
                if idx < n + keyframe_original_idx[0]:
                    matrix = vec7_to_4x4(cam_data_seq[idx])
                    gt_absolute_poses.append(torch.as_tensor(matrix))
                else:
                    gt_absolute_poses.append(torch.eye(4)[:3, :])
                
            if idx + 1 < n + keyframe_original_idx[0]:
                cam_prev = cam_data_seq[idx]
                cam_next = cam_data_seq[idx + 1]  # SpatialVid: every 1 frame
                relative_cam = compute_relative_pose_matrix(cam_prev, cam_next)
                relative_cams.append(torch.as_tensor(relative_cam[:3, :]))
                matrix = vec7_to_4x4(cam_next)
                gt_absolute_poses.append(torch.as_tensor(matrix))
            else:
                break
        
        if len(relative_cams) <= 1:
            return None, None

        pose_embedding = torch.stack(relative_cams, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        
        random_pose_embedding, random_absolute_poses = generate_random_camera_poses(m, gt_absolute_poses[-1])
        gt_absolute_poses = gt_absolute_poses + random_absolute_poses
        pose_embedding = torch.cat([pose_embedding, random_pose_embedding], dim=0)
        pose_embedding = pose_embedding.to(torch.bfloat16)
        
        return pose_embedding, gt_absolute_poses

    def prepare_framepack_inputs(self, full_latents, segment_info):
        """🔧 Prepare FramePack-style multi-scale inputs - SpatialVid version"""
        # 🔧 Fix: Handle 4D input [C, T, H, W], add batch dimension
        if len(full_latents.shape) == 4:
            full_latents = full_latents.unsqueeze(0)  # [C, T, H, W] -> [1, C, T, H, W]
            B, C, T, H, W = full_latents.shape
        else:
            B, C, T, H, W = full_latents.shape
        
        # Main latents (for denoising prediction)
        latent_indices = segment_info['latent_indices']
        main_latents = full_latents[:, :, latent_indices, :, :]
        
        # 🔧 1x condition frames (start frame + last 1 frame)
        clean_latent_indices = segment_info['clean_latent_indices']
        clean_latents = full_latents[:, :, clean_latent_indices, :, :]
        
        # 🔧 4x condition frames - always 16 frames, use actual indices + 0 padding
        clean_latent_4x_indices = segment_info['clean_latent_4x_indices']
        
        # Create fixed-length 16 latents, initialize to 0
        clean_latents_4x = torch.zeros(B, C, 16, H, W, dtype=full_latents.dtype)
        clean_latent_4x_indices_final = torch.full((16,), -1, dtype=torch.long)  # -1 indicates padding
        
        # 🔧 Fix: Check if there are valid 4x indices
        if len(clean_latent_4x_indices) > 0:
            actual_4x_frames = len(clean_latent_4x_indices)
            # Fill from back to front, ensuring newest frames are at the end
            start_pos = max(0, 16 - actual_4x_frames)
            end_pos = 16
            actual_start = max(0, actual_4x_frames - 16)  # If more than 16 frames, take only last 16
            
            clean_latents_4x[:, :, start_pos:end_pos, :, :] = full_latents[:, :, clean_latent_4x_indices[actual_start:], :, :]
            clean_latent_4x_indices_final[start_pos:end_pos] = clean_latent_4x_indices[actual_start:]
        
        # 🔧 2x condition frames - always 2 frames, use actual indices + 0 padding
        clean_latent_2x_indices = segment_info['clean_latent_2x_indices']
        
        # Create fixed-length 2 latents, initialize to 0
        clean_latents_2x = torch.zeros(B, C, 2, H, W, dtype=full_latents.dtype)
        clean_latent_2x_indices_final = torch.full((2,), -1, dtype=torch.long)  # -1 indicates padding
        
        # 🔧 Fix: Check if there are valid 2x indices
        if len(clean_latent_2x_indices) > 0:
            actual_2x_frames = len(clean_latent_2x_indices)
            # Fill from back to front, ensuring newest frames are at the end
            start_pos = max(0, 2 - actual_2x_frames)
            end_pos = 2
            actual_start = max(0, actual_2x_frames - 2)  # If more than 2 frames, take only last 2
            
            clean_latents_2x[:, :, start_pos:end_pos, :, :] = full_latents[:, :, clean_latent_2x_indices[actual_start:], :, :]
            clean_latent_2x_indices_final[start_pos:end_pos] = clean_latent_2x_indices[actual_start:]
        
        # 🔧 Remove added batch dimension, return original format
        if B == 1:
            main_latents = main_latents.squeeze(0)  # [1, C, T, H, W] -> [C, T, H, W]
            clean_latents = clean_latents.squeeze(0)
            clean_latents_2x = clean_latents_2x.squeeze(0)
            clean_latents_4x = clean_latents_4x.squeeze(0)
        
        return {
            'latents': main_latents,
            'clean_latents': clean_latents,
            'clean_latents_2x': clean_latents_2x,
            'clean_latents_4x': clean_latents_4x,
            'latent_indices': segment_info['latent_indices'],
            'clean_latent_indices': segment_info['clean_latent_indices'],
            'clean_latent_2x_indices': clean_latent_2x_indices_final,
            'clean_latent_4x_indices': clean_latent_4x_indices_final,
        }

    def __getitem__(self, index):
        while True:
            try:
                # choose a scene randomly
                scene_dir = random.choice(self.scene_dirs)
                # print(f"Loading scenes: {scene_dir}", flush=True)
                
                # Load encoded video data
                encoded_data = torch.load(
                    # os.path.join(scene_dir, "encoded_video.pth"),
                    scene_dir,
                    weights_only=False,
                    map_location="cpu"
                )
                
                # 🔧 Verify if latent frame count matches expected
                full_latents = encoded_data['latents']  # [C, T, H, W]
                cam_data = encoded_data['cam_emb']
                actual_latent_frames = full_latents.shape[1]
                
                # Dynamically select segment
                segment_info = self.select_dynamic_segment_framepack(full_latents)
                if segment_info is None:
                    continue
                
                # Create pose embeddings - SpatialVid version
                all_camera_embeddings, gt_absolute_poses = self.create_random_pose_embeddings(cam_data, segment_info)
                if all_camera_embeddings is None:
                    continue
                
                # 🔧 Prepare FramePack-style multi-scale inputs
                framepack_inputs = self.prepare_framepack_inputs(full_latents, segment_info)
                
                n = segment_info["condition_frames"]
                m = segment_info['target_frames']

                # 🔧 Process camera embedding with mask
                mask = torch.zeros(n+m, dtype=torch.float32)
                mask[:n] = 1.0  # Mark condition frames as 1
                mask = mask.view(-1, 1)
                
                # gt_relative_poses = gt_relative_poses[n:]  # Only keep relative poses of target frames
                gt_absolute_poses = gt_absolute_poses[n:]  # Only keep absolute poses of target frames
                # gt_absolute_poses = gt_absolute_poses[n+1:]

                # Add mask to camera embeddings
                camera_with_mask = torch.cat([all_camera_embeddings, mask], dim=1)
                
                # Extract prompt text
                prompt_text = self.scene2prompt[scene_dir]
                
                result = {
                    # 🔧 FramePack-style multi-scale inputs
                    "latents": framepack_inputs['latents'],  # Main prediction target
                    "clean_latents": framepack_inputs['clean_latents'],  # Condition frames
                    "clean_latents_2x": framepack_inputs['clean_latents_2x'],
                    "clean_latents_4x": framepack_inputs['clean_latents_4x'],
                    "latent_indices": framepack_inputs['latent_indices'],
                    "clean_latent_indices": framepack_inputs['clean_latent_indices'],
                    "clean_latent_2x_indices": framepack_inputs['clean_latent_2x_indices'],
                    "clean_latent_4x_indices": framepack_inputs['clean_latent_4x_indices'],
                    
                    # 🔧 Pass camera embeddings with mask directly
                    "camera": camera_with_mask,  # Camera embeddings for all frames (with mask) N * 13
                    "gt_absolute_poses": gt_absolute_poses, # Absolute pose matrices for target frames
                    
                    "prompt_emb": encoded_data["prompt_emb"],
                    "prompt_text": prompt_text,
                    "image_emb": encoded_data.get("image_emb", {}),
                    
                    "condition_frames": n,  # Compressed frame count
                    "target_frames": m,  # Compressed frame count
                    "scene_name": os.path.basename(scene_dir),
                    "dataset_name": "spatialvid",
                    # 🔧 New: record original frame count for debugging
                    "original_condition_frames": segment_info['original_condition_frames'],
                    "original_target_frames": segment_info['original_target_frames'],
                }
                
                return result
                
            except Exception as e:
                print(f"Error loading sample: {e}")
                traceback.print_exc()
                continue
    
    def __len__(self):
        return self.steps_per_epoch
