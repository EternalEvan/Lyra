from pathlib import Path
from typing import Optional
import torch
import numpy as np
from torchvision.transforms import v2
from PIL import Image
import imageio
from einops import rearrange

from diffsynth import WanVideoAstraPipeline

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
class InlineVideoEncoder:

    def __init__(self, pipe: WanVideoAstraPipeline, device="cuda"):
        self.device = getattr(pipe, "device", device)
        self.tiler_kwargs = {"tiled": True, "tile_size": (34, 34), "tile_stride": (18, 16)}
        self.frame_process = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.pipe = pipe

    @staticmethod
    def _crop_and_resize(image: Image.Image) -> Image.Image:
        target_w, target_h = 832, 480
        return v2.functional.resize(
            image,
            (round(target_h), round(target_w)),
            interpolation=v2.InterpolationMode.BILINEAR,
        )

    def preprocess_frame(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = self._crop_and_resize(image)
        return self.frame_process(image)

    def load_video_frames(self, video_path: Path) -> Optional[torch.Tensor]:
        reader = imageio.get_reader(str(video_path))
        frames = []
        for frame_data in reader:
            frame = Image.fromarray(frame_data)
            frames.append(self.preprocess_frame(frame))
        reader.close()

        if not frames:
            return None

        frames = torch.stack(frames, dim=0)
        return rearrange(frames, "T C H W -> C T H W")

    def encode_frames_to_latents(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            latents = self.pipe.encode_video(frames, **self.tiler_kwargs)[0]

        if latents.dim() == 5 and latents.shape[0] == 1:
            latents = latents.squeeze(0)
        return latents.cpu()
    
def image_to_frame_stack(
    image_path: Path, 
    encoder: InlineVideoEncoder, 
    repeat_count: int = 10
) -> torch.Tensor:
    """Repeat a single image into a tensor with specified number of frames, shape [C, T, H, W]"""
    if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    image = Image.open(str(image_path))
    frame = encoder.preprocess_frame(image)
    frames = torch.stack([frame for _ in range(repeat_count)], dim=0)
    return rearrange(frames, "T C H W -> C T H W")

#==========================#
# Camera Motion Primitives
#==========================#
class CameraMotionBuilder:
    """基础相机运动模式构建器，提供原子运动操作"""
    
    @staticmethod
    def zero_motion():
        """零运动（静止）"""
        return np.eye(4, dtype=np.float32)
    
    @staticmethod
    def forward(forward_speed=0.03):
        """前进运动
        Args:
            forward_speed: 前进速度（负值表示向前进）
        """
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = -forward_speed
        return pose
    
    @staticmethod
    def rotate(yaw_per_frame=0.03, forward_speed=0.00):
        """旋转运动
        Args:
            yaw_per_frame: 每帧的偏航角（正值向左转，负值向右转）
            forward_speed: 前进速度
        """
        cos_yaw = np.cos(yaw_per_frame)
        sin_yaw = np.sin(yaw_per_frame)
        
        pose = np.eye(4, dtype=np.float32)
        pose[0, 0] = cos_yaw
        pose[0, 2] = sin_yaw
        pose[2, 0] = -sin_yaw
        pose[2, 2] = cos_yaw
        pose[2, 3] = -forward_speed
        return pose
    
    @staticmethod
    def rotate_left(yaw_per_frame=0.03, forward_speed=0.00):
        """左转"""
        return CameraMotionBuilder.rotate(yaw_per_frame=yaw_per_frame, forward_speed=forward_speed)
    
    @staticmethod
    def rotate_right(yaw_per_frame=0.03, forward_speed=0.00):
        """右转"""
        return CameraMotionBuilder.rotate(yaw_per_frame=-yaw_per_frame, forward_speed=forward_speed)
    
    @staticmethod
    def sideways(drift_x=0.01, forward_speed=0.00):
        """横向漂移
        Args:
            drift_x: 横向漂移量（负值向左，正值向右）
            forward_speed: 前进速度
        """
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = -forward_speed
        pose[0, 3] = drift_x
        return pose


class CameraTrajectory:
    """相机轨迹定义，支持分阶段运动"""
    
    def __init__(self, name="custom"):
        self.name = name
        self.stages = []  # 存储运动阶段: (frame_count, motion_function)
    
    def add_stage(self, frame_count, motion_func):
        """添加一个运动阶段
        Args:
            frame_count: 该阶段的帧数
            motion_func: 返回4x4位姿矩阵的函数，可以是CameraMotionBuilder的方法
        """
        self.stages.append((frame_count, motion_func))
        return self
    
    def generate_poses(self, total_frames, condition_frames):
        """生成完整的运动序列
        Args:
            total_frames: 总帧数
            condition_frames: 条件帧数
        Returns:
            poses: 4x4位姿矩阵列表
        """
        poses = []
        current_stage_idx = 0
        frames_in_current_stage = 0
        
        for i in range(total_frames):
            # 条件帧使用零运动
            if i < condition_frames:
                poses.append(CameraMotionBuilder.zero_motion())
            else:
                # 检查是否需要进入下一阶段
                gen_frame_idx = i - condition_frames
                
                # 累积计算当前阶段
                while current_stage_idx < len(self.stages):
                    stage_frames, stage_func = self.stages[current_stage_idx]
                    if frames_in_current_stage < stage_frames:
                        # 仍在当前阶段
                        poses.append(stage_func())
                        frames_in_current_stage += 1
                        break
                    else:
                        # 进入下一阶段
                        frames_in_current_stage = 0
                        current_stage_idx += 1
                        if current_stage_idx < len(self.stages):
                            # 使用新阶段的运动函数
                            poses.append(stage_func())
                            frames_in_current_stage += 1
                            break
                        else:
                            # 所有阶段都结束了，使用静止
                            poses.append(CameraMotionBuilder.zero_motion())
                            break
        
        return poses