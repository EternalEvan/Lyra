#This code file is from [https://github.com/hao-ai-lab/FastVideo], which is licensed under Apache License 2.0.


import math
import random
import numpy as np
from collections import Counter
from typing import List, Optional

import decord
import torch
import torch.utils
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import Sampler

IMG_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f"{self.__class__.__name__}("
                    f"sr={self.sr},"
                    f"num_threads={self.num_threads})")
        return repr_str


def pad_to_multiple(number, ds_stride):
    remainder = number % ds_stride
    if remainder == 0:
        return number
    else:
        padding = ds_stride - remainder
        return number + padding


# TODO
class Collate:

    def __init__(self, args):
        self.batch_size = args.train_batch_size
        self.group_frame = args.group_frame
        self.group_resolution = args.group_resolution

        self.max_height = args.max_height
        self.max_width = args.max_width
        self.ae_stride = args.ae_stride

        self.ae_stride_t = args.ae_stride_t
        self.ae_stride_thw = (self.ae_stride_t, self.ae_stride, self.ae_stride)

        self.patch_size = args.patch_size
        self.patch_size_t = args.patch_size_t

        self.num_frames = args.num_frames
        self.use_image_num = args.use_image_num
        self.max_thw = (self.num_frames, self.max_height, self.max_width)

    def package(self, batch):
        batch_tubes = [i["pixel_values"] for i in batch]  # b [c t h w]
        input_ids = [i["input_ids"] for i in batch]  # b [1 l]
        cond_mask = [i["cond_mask"] for i in batch]  # b [1 l]
        return batch_tubes, input_ids, cond_mask

    def __call__(self, batch):
        batch_tubes, input_ids, cond_mask = self.package(batch)

        ds_stride = self.ae_stride * self.patch_size
        t_ds_stride = self.ae_stride_t * self.patch_size_t

        pad_batch_tubes, attention_mask, input_ids, cond_mask = self.process(
            batch_tubes,
            input_ids,
            cond_mask,
            t_ds_stride,
            ds_stride,
            self.max_thw,
            self.ae_stride_thw,
        )
        assert not torch.any(
            torch.isnan(pad_batch_tubes)), "after pad_batch_tubes"
        return pad_batch_tubes, attention_mask, input_ids, cond_mask

    def process(
        self,
        batch_tubes,
        input_ids,
        cond_mask,
        t_ds_stride,
        ds_stride,
        max_thw,
        ae_stride_thw,
    ):
        # pad to max multiple of ds_stride
        batch_input_size = [i.shape
                            for i in batch_tubes]  # [(c t h w), (c t h w)]
        assert len(batch_input_size) == self.batch_size
        if self.group_frame or self.group_resolution or self.batch_size == 1:  #
            len_each_batch = batch_input_size
            idx_length_dict = dict(
                [*zip(list(range(self.batch_size)), len_each_batch)])
            count_dict = Counter(len_each_batch)
            if len(count_dict) != 1:
                sorted_by_value = sorted(count_dict.items(),
                                         key=lambda item: item[1])
                pick_length = sorted_by_value[-1][0]  # the highest frequency
                candidate_batch = [
                    idx for idx, length in idx_length_dict.items()
                    if length == pick_length
                ]
                random_select_batch = [
                    random.choice(candidate_batch)
                    for _ in range(len(len_each_batch) - len(candidate_batch))
                ]
                print(
                    batch_input_size,
                    idx_length_dict,
                    count_dict,
                    sorted_by_value,
                    pick_length,
                    candidate_batch,
                    random_select_batch,
                )
                pick_idx = candidate_batch + random_select_batch

                batch_tubes = [batch_tubes[i] for i in pick_idx]
                batch_input_size = [i.shape for i in batch_tubes
                                    ]  # [(c t h w), (c t h w)]
                input_ids = [input_ids[i] for i in pick_idx]  # b [1, l]
                cond_mask = [cond_mask[i] for i in pick_idx]  # b [1, l]

            for i in range(1, self.batch_size):
                assert batch_input_size[0] == batch_input_size[i]
            max_t = max([i[1] for i in batch_input_size])
            max_h = max([i[2] for i in batch_input_size])
            max_w = max([i[3] for i in batch_input_size])
        else:
            max_t, max_h, max_w = max_thw
        pad_max_t, pad_max_h, pad_max_w = (
            pad_to_multiple(max_t - 1 + self.ae_stride_t, t_ds_stride),
            pad_to_multiple(max_h, ds_stride),
            pad_to_multiple(max_w, ds_stride),
        )
        pad_max_t = pad_max_t + 1 - self.ae_stride_t
        each_pad_t_h_w = [[
            pad_max_t - i.shape[1], pad_max_h - i.shape[2],
            pad_max_w - i.shape[3]
        ] for i in batch_tubes]
        pad_batch_tubes = [
            F.pad(im, (0, pad_w, 0, pad_h, 0, pad_t), value=0)
            for (pad_t, pad_h, pad_w), im in zip(each_pad_t_h_w, batch_tubes)
        ]
        pad_batch_tubes = torch.stack(pad_batch_tubes, dim=0)

        max_tube_size = [pad_max_t, pad_max_h, pad_max_w]
        max_latent_size = [
            ((max_tube_size[0] - 1) // ae_stride_thw[0] + 1),
            max_tube_size[1] // ae_stride_thw[1],
            max_tube_size[2] // ae_stride_thw[2],
        ]
        valid_latent_size = [[
            int(math.ceil((i[1] - 1) / ae_stride_thw[0])) + 1,
            int(math.ceil(i[2] / ae_stride_thw[1])),
            int(math.ceil(i[3] / ae_stride_thw[2])),
        ] for i in batch_input_size]
        attention_mask = [
            F.pad(
                torch.ones(i, dtype=pad_batch_tubes.dtype),
                (
                    0,
                    max_latent_size[2] - i[2],
                    0,
                    max_latent_size[1] - i[1],
                    0,
                    max_latent_size[0] - i[0],
                ),
                value=0,
            ) for i in valid_latent_size
        ]
        attention_mask = torch.stack(attention_mask)  # b t h w
        if self.batch_size == 1 or self.group_frame or self.group_resolution:
            assert torch.all(attention_mask.bool())

        input_ids = torch.stack(input_ids)  # b 1 l
        cond_mask = torch.stack(cond_mask)  # b 1 l

        return pad_batch_tubes, attention_mask, input_ids, cond_mask


def split_to_even_chunks(indices, lengths, num_chunks, batch_size):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        chunks = [indices[i::num_chunks] for i in range(num_chunks)]
    else:
        num_indices_per_chunk = len(indices) // num_chunks

        chunks = [[] for _ in range(num_chunks)]
        chunks_lengths = [0 for _ in range(num_chunks)]
        for index in indices:
            shortest_chunk = chunks_lengths.index(min(chunks_lengths))
            chunks[shortest_chunk].append(index)
            chunks_lengths[shortest_chunk] += lengths[index]
            if len(chunks[shortest_chunk]) == num_indices_per_chunk:
                chunks_lengths[shortest_chunk] = float("inf")
    # return chunks

    pad_chunks = []
    for idx, chunk in enumerate(chunks):
        if batch_size != len(chunk):
            assert batch_size > len(chunk)
            if len(chunk) != 0:
                chunk = chunk + [
                    random.choice(chunk)
                    for _ in range(batch_size - len(chunk))
                ]
            else:
                chunk = random.choice(pad_chunks)
                print(chunks[idx], "->", chunk)
        pad_chunks.append(chunk)
    return pad_chunks


def group_frame_fun(indices, lengths):
    # sort by num_frames
    indices.sort(key=lambda i: lengths[i], reverse=True)
    return indices


def megabatch_frame_alignment(megabatches, lengths):
    aligned_magabatches = []
    for _, megabatch in enumerate(megabatches):
        assert len(megabatch) != 0
        len_each_megabatch = [lengths[i] for i in megabatch]
        idx_length_dict = dict([*zip(megabatch, len_each_megabatch)])
        count_dict = Counter(len_each_megabatch)

        # mixed frame length, align megabatch inside
        if len(count_dict) != 1:
            sorted_by_value = sorted(count_dict.items(),
                                     key=lambda item: item[1])
            pick_length = sorted_by_value[-1][0]  # the highest frequency
            candidate_batch = [
                idx for idx, length in idx_length_dict.items()
                if length == pick_length
            ]
            random_select_batch = [
                random.choice(candidate_batch)
                for i in range(len(idx_length_dict) - len(candidate_batch))
            ]
            aligned_magabatch = candidate_batch + random_select_batch
            aligned_magabatches.append(aligned_magabatch)
        # already aligned megabatches
        else:
            aligned_magabatches.append(megabatch)

    return aligned_magabatches


def get_length_grouped_indices(
    lengths,
    batch_size,
    world_size,
    generator=None,
    group_frame=False,
    group_resolution=False,
    seed=42,
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    if generator is None:
        generator = torch.Generator().manual_seed(
            seed)  # every rank will generate a fixed order but random index

    indices = torch.randperm(len(lengths), generator=generator).tolist()

    # sort dataset according to frame
    indices = group_frame_fun(indices, lengths)

    # chunk dataset to megabatches
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i:i + megabatch_size]
        for i in range(0, len(lengths), megabatch_size)
    ]

    # make sure the length in each magabatch is align with each other
    megabatches = megabatch_frame_alignment(megabatches, lengths)

    # aplit aligned megabatch into batches
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size, batch_size)
        for megabatch in megabatches
    ]

    # random megabatches to do video-image mix training
    indices = torch.randperm(len(megabatches), generator=generator).tolist()
    shuffled_megabatches = [megabatches[i] for i in indices]

    # expand indices and return
    return [
        i for megabatch in shuffled_megabatches for batch in megabatch
        for i in batch
    ]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        rank: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        group_frame=False,
        group_resolution=False,
        generator=None,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.lengths = lengths
        self.group_frame = group_frame
        self.group_resolution = group_resolution
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(
            self.lengths,
            self.batch_size,
            self.world_size,
            group_frame=self.group_frame,
            group_resolution=self.group_resolution,
            generator=self.generator,
        )

        def distributed_sampler(lst, rank, batch_size, world_size):
            result = []
            index = rank * batch_size
            while index < len(lst):
                result.extend(lst[index:index + batch_size])
                index += batch_size * world_size
            return result

        indices = distributed_sampler(indices, self.rank, self.batch_size,
                                      self.world_size)
        return iter(indices)

def _create_rotation_x(angle):
    """Create rotation matrix around X-axis (pitch)"""
    c = torch.cos(angle).item()
    s = torch.sin(angle).item()
    return torch.tensor([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)


def _create_rotation_y(angle):
    """Create rotation matrix around Y-axis (yaw)"""
    c = torch.cos(angle).item()
    s = torch.sin(angle).item()
    return torch.tensor([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)


def _create_translation(x=0, y=0, z=0):
    """Create translation matrix"""
    return torch.tensor([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ], dtype=torch.float32)


def _create_smooth_transform(direction, angle, distance):
    """
    Create smooth camera motion with combined rotation and translation.
    Simulates curved trajectories with slight radial drift.
    
    Args:
        direction: Direction of motion ('up', 'down', 'left', 'right')
        angle: Rotation angle in radians
        distance: Forward distance per frame
    Returns:
        4x4 transformation matrix
    """
    pose = np.eye(4, dtype=np.float32)
    
    if direction == 'up':
        # Pitch up: rotate around X-axis and move forward
        cos_pitch = np.cos(angle)
        sin_pitch = np.sin(angle)
        pose[1, 1] = cos_pitch
        pose[1, 2] = sin_pitch
        pose[2, 1] = -sin_pitch
        pose[2, 2] = cos_pitch
        pose[2, 3] = -distance  # Move forward (negative Z)
        
    elif direction == 'down':
        # Pitch down: rotate around X-axis and move forward
        cos_pitch = np.cos(-angle)
        sin_pitch = np.sin(-angle)
        pose[1, 1] = cos_pitch
        pose[1, 2] = sin_pitch
        pose[2, 1] = -sin_pitch
        pose[2, 2] = cos_pitch
        pose[2, 3] = -distance
        
    elif direction == 'left':
        # Turn left: rotate around Y-axis, move forward with slight inward drift
        cos_yaw = np.cos(angle)
        sin_yaw = np.sin(angle)
        pose[0, 0] = cos_yaw
        pose[0, 2] = sin_yaw
        pose[2, 0] = -sin_yaw
        pose[2, 2] = cos_yaw
        pose[2, 3] = -distance  # Move forward
        pose[0, 3] = -0.002  # Slight radial drift inward
        
    elif direction == 'right':
        # Turn right: rotate around Y-axis, move forward with slight inward drift
        cos_yaw = np.cos(-angle)
        sin_yaw = np.sin(-angle)
        pose[0, 0] = cos_yaw
        pose[0, 2] = sin_yaw
        pose[2, 0] = -sin_yaw
        pose[2, 2] = cos_yaw
        pose[2, 3] = -distance  # Move forward
        pose[0, 3] = 0.002  # Slight radial drift inward
    
    return torch.tensor(pose, dtype=torch.float32)


def generate_random_camera_poses(num_frames, last_absolute_pose=None):
    """
    Generate random relative camera poses and absolute pose matrices.
    
    Supports three types of camera motion:
    - Pure rotation: pitch (up/down) or yaw (left/right)
    - Pure translation: move along X or Y axis
    - Smooth motion: combined rotation + translation with curved trajectory
    
    Args:
        num_frames: Number of frames to generate
        last_absolute_pose: Last absolute pose matrix (4x4 tensor), if None start from identity
    
    Returns:
        relative_poses_tensor: Tensor of shape (N+1, 12), each row is flattened 3x4 relative pose matrix
        absolute_poses: List of N*4*4 tensors, absolute pose matrices
    """
    # Motion parameters
    DISTANCE = 0.05
    ANGLE_DEG = 3
    angle_rad = torch.tensor(np.radians(ANGLE_DEG), dtype=torch.float32)
    
    # Define available movement types
    movement_types = {
        'rotate_up': ('rotation', 'up', -angle_rad),
        'rotate_down': ('rotation', 'down', angle_rad),
        'rotate_left': ('rotation', 'left', angle_rad),
        'rotate_right': ('rotation', 'right', -angle_rad),
        'translate_left': ('translation', 'left', -DISTANCE),
        'translate_right': ('translation', 'right', DISTANCE),
        'translate_forward': ('translation', 'forward', -DISTANCE),
        'translate_backward': ('translation', 'backward', DISTANCE),
        'smooth_up': ('smooth', 'up', angle_rad, DISTANCE / 2),
        'smooth_down': ('smooth', 'down', -angle_rad, DISTANCE / 2),
        'smooth_left': ('smooth', 'left', angle_rad, DISTANCE / 2),
        'smooth_right': ('smooth', 'right', -angle_rad, DISTANCE / 2),
    }
    
    # Randomly select movement type
    movement_type = random.choice(list(movement_types.keys()))
    motion_info = movement_types[movement_type]
    
    # Initialize absolute pose (identity matrix if not provided)
    current_pose = last_absolute_pose.clone() if last_absolute_pose is not None else torch.eye(4, dtype=torch.float32)
    
    relative_poses_list = []
    absolute_poses = []
    
    for _ in range(num_frames):
        # Generate relative transformation based on movement type
        motion_type = motion_info[0]
        
        if motion_type == 'rotation':
            # Pure rotation: pitch (X-axis) or yaw (Y-axis)
            direction, angle = motion_info[1], motion_info[2]
            if direction in ['up', 'down']:
                relative_transform = _create_rotation_x(angle)
            else:  # left or right
                relative_transform = _create_rotation_y(angle)
                
        elif motion_type == 'translation':
            # Pure translation
            direction, distance = motion_info[1], motion_info[2]
            if direction == 'left':
                relative_transform = _create_translation(x=-abs(distance))
            elif direction == 'right':
                relative_transform = _create_translation(x=abs(distance))
            elif direction == 'forward':
                relative_transform = _create_translation(z=-abs(distance))
            elif direction == 'backward':
                relative_transform = _create_translation(z=abs(distance))
                
        elif motion_type == 'smooth':
            # Smooth motion: combined rotation and translation
            direction, angle, distance = motion_info[1], motion_info[2], motion_info[3]
            relative_transform = _create_smooth_transform(direction, angle, distance)
        
        # Extract 3x4 relative pose matrix (remove last row [0,0,0,1])
        relative_pose_3x4 = relative_transform[:3, :]
        relative_poses_list.append(relative_pose_3x4.flatten())
        
        # Update absolute pose
        current_pose = current_pose @ relative_transform
        absolute_poses.append(current_pose.clone())
    
    # Add zero relative pose for last frame
    zero_relative = torch.eye(4, dtype=torch.float32)[:3, :].flatten()
    relative_poses_list.append(zero_relative)
    
    # Stack relative poses into tensor
    relative_poses_tensor = torch.stack(relative_poses_list)
    
    return relative_poses_tensor, absolute_poses
