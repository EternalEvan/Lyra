import torch
import numpy as np
import imageio.v2 as imageio
import os
import shutil
from scipy.spatial.transform import Rotation
import argparse

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images, load_and_preprocess_tensors
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

class CameraAlignmentReward:
    """
    一个用于计算 VGGT 预测的相机轨迹与地面真实轨迹之间对齐奖励的类。
    更新版：使用四元数计算旋转误差，并使用 Procrustes 分析对齐轨迹尺度。
    """

    def __init__(self, model_name="facebook/VGGT-1B", cache_dir="/data1/zyx/vggt", rank=None, device=None):
        """
        初始化模型和设备。

        Args:
            model_name (str): 要加载的预训练模型的名称。
            cache_dir (str): 存放预训练模型的缓存目录。
            device (str, optional): 运行模型的设备 ('cuda' 或 'cpu')。如果为 None，则自动检测。
        """
        print("正在初始化 CameraAlignmentReward...")
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if rank is None:
            self.rank = 0
        else:
            self.rank = rank
        
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        print(f"使用设备: {self.device}, 数据类型: {self.dtype}")
        print("正在加载 VGGT 模型...")
        self.model = VGGT.from_pretrained(model_name, cache_dir=cache_dir).to(self.device).eval()
        print(f"模型加载完成，RANK: {rank}。")

    def _predict_extrinsics(self, video_tensor: torch.Tensor) -> np.ndarray:
        """
        从视频文件中预测相机外参。
        """
        temp_frames_dir = f"temp_frames_for_reward_{self.rank}"
        os.makedirs(temp_frames_dir, exist_ok=True)

        video_tensor = video_tensor.to(self.device)
        # video_tensor = video_tensor.unsqueeze(0)

        print(f"正在预处理视频帧...")
        images = load_and_preprocess_tensors(video_tensor).to(self.device)
        images = images.unsqueeze(0)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=self.dtype):
                print("正在运行模型推理以预测相机参数...")
                aggregated_tokens_list, _ = self.model.aggregator(images)
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                compressed_pose_enc = pose_enc[:, 0::4, :]
                # compressed_pose_enc = pose_enc
                extrinsics, _ = pose_encoding_to_extri_intri(compressed_pose_enc, images.shape[-2:])
                # 加入新的一行 [0 0 0 1] 以形成完整的 4x4 外参矩阵
                extrinsics = torch.cat([extrinsics,
                                        torch.tensor([0, 0, 0, 1], device=self.device, dtype=extrinsics.dtype)
                                        .view(1, 1, 1, 4).repeat(1, extrinsics.shape[1], 1, 1)], dim=2)
        return extrinsics.squeeze(0).cpu().numpy()

    @staticmethod
    def _convert_to_relative_poses(extrinsics: np.ndarray) -> np.ndarray:
        """
        将绝对外参矩阵转换为相对于第一帧的相对位姿。
        """
        print(extrinsics.shape)
        first_frame_inv = np.linalg.inv(extrinsics[0])
        relative_poses = np.array([first_frame_inv @ p for p in extrinsics])
        return relative_poses
    
    @staticmethod
    def _align_scale(gt_translations: np.ndarray, pred_translations: np.ndarray, 
                    eps: float = 1e-6) -> tuple[np.ndarray, float]:
        """
        使用 Procrustes 分析对齐预测轨迹的尺度。
        
        Args:
            gt_translations: 真实轨迹 [n, 3]
            pred_translations: 预测轨迹 [n, 3]
            eps: 数值稳定性阈值
        
        Returns:
            scaled_pred_translations: 尺度对齐后的预测轨迹
            scale: 应用的尺度因子
        """
        # 输入验证
        if gt_translations.shape != pred_translations.shape:
            raise ValueError("gt_translations and pred_translations must have the same shape")
        
        n_frames = gt_translations.shape[0]
        if n_frames < 2:
            # 单帧情况下无法有意义地计算尺度
            return pred_translations, 1.0
        
        # 检查全零情况
        gt_norm = np.linalg.norm(gt_translations, axis=1).max()
        pred_norm = np.linalg.norm(pred_translations, axis=1).max()
        
        if gt_norm < eps:
            # 两者都接近零，不需要缩放
            return pred_translations, 1.0
        
        if pred_norm < eps:
            # 预测轨迹接近全零
            return np.zeros_like(pred_translations), 0.0
        
        # 去中心化
        gt_mean = gt_translations.mean(axis=0)
        pred_mean = pred_translations.mean(axis=0)
        
        gt_centered = gt_translations - gt_mean
        pred_centered = pred_translations - pred_mean
        
        # 计算尺度因子
        numerator = np.sum(gt_centered * pred_centered)
        denominator = np.sum(pred_centered**2)
        
        if denominator < eps:
            # 预测轨迹变化太小，无法可靠计算尺度
            return pred_translations, 1.0
        
        scale = abs(numerator / denominator)
        
        # 应用尺度
        scaled_pred_translations = pred_translations * scale
        
        return scaled_pred_translations, scale
    
    @staticmethod
    def _is_near_identity_rotation(rotations: Rotation, threshold_deg: float = 1.0) -> bool:
        """Judge whether every rotation in the batch is close to identity."""
        angles_deg = np.rad2deg(rotations.magnitude())
        return np.all(angles_deg < threshold_deg)
    
    def calculate_reward(self, video_tensor: torch.Tensor, gt_extrinsics: np.ndarray, rot_weight: float = 0.7, trans_weight: float = 0.3) -> dict:
        """
        计算预测位姿和真实位姿之间的对齐奖励。

        Args:
            video_path (str): 视频文件的路径。
            gt_extrinsics (np.ndarray): 地面真实相机外参矩阵，形状为 (N, 4, 4)。
            rot_weight (float): 旋转奖励在总奖励中的权重。
            trans_weight (float): 平移奖励在总奖励中的权重。

        Returns:
            dict: 包含详细误差和最终奖励的字典。
        """
        print("开始计算对齐奖励...")
        pred_extrinsics = self._predict_extrinsics(video_tensor)
        print(len(pred_extrinsics))
        
        num_frames = len(pred_extrinsics)
        if num_frames != len(gt_extrinsics):
            raise ValueError(f"预测帧数 ({num_frames}) 与 GT 帧数 ({len(gt_extrinsics)}) 不匹配。")

        relative_pred_poses = self._convert_to_relative_poses(pred_extrinsics)
        relative_gt_poses = self._convert_to_relative_poses(gt_extrinsics)

        # --- 旋转误差计算 (使用四元数) ---
        pred_rotations = Rotation.from_matrix(relative_pred_poses[:, :3, :3])
        gt_rotations = Rotation.from_matrix(relative_gt_poses[:, :3, :3])
        
        # 计算旋转差异: R_err = R_gt * R_pred^{-1}
        error_rotations = gt_rotations * pred_rotations.inv()
        
        # 差异旋转的角度 (角距离)
        rot_errors_rad = error_rotations.magnitude()
        mean_rot_error_deg = np.rad2deg(np.mean(rot_errors_rad))

        # --- 平移误差计算 (先对齐尺度) ---
        pred_translations = relative_pred_poses[:, :3, 3]
        gt_translations = relative_gt_poses[:, :3, 3]

        # 1. 首先，对齐预测轨迹的尺度以匹配 GT 轨迹
        scaled_pred_translations, scale_factor = self._align_scale(gt_translations, pred_translations)
        
        # 2. 计算对齐后的平移误差
        trans_errors = np.linalg.norm(scaled_pred_translations - gt_translations, axis=1)
        mean_trans_error = np.mean(trans_errors)

        # --- 将误差转换为奖励 (0 到 1) ---
        k_rot = 0.25  # 旋转误差每增加4度，奖励大约衰减到 e^-1
        k_trans = 20.0 # 平移误差每增加0.05个单位，奖励大约衰减到 e^-1
        
        rotation_reward = np.exp(-k_rot * mean_rot_error_deg)
        translation_reward = np.exp(-k_trans * mean_trans_error)
        
        if self._is_near_identity_rotation(gt_rotations):
            print("输入的旋转接近恒等旋转，将调整奖励的权重以更侧重平移误差")
            rot_weight = 0.3
            trans_weight = 0.7
        else:
            print("输入的旋转具有显著变化，保持默认的奖励权重，更侧重旋转误差")
            rot_weight = 0.7
            trans_weight = 0.3

        total_reward = (rot_weight * rotation_reward + trans_weight * translation_reward) / (rot_weight + trans_weight)

        print("奖励计算完成。")

        return {
            "total_reward": total_reward,
            "rotation_reward": rotation_reward,
            "translation_reward": translation_reward,
            "mean_rotation_error_degrees": mean_rot_error_deg,
            "mean_translation_error": mean_trans_error,
            "translation_scale_factor": scale_factor
        }

# --- 使用示例 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Camera Alignment Reward Calculation")
    parser.add_argument('--video_paths', '-f',
                        nargs='+',
                        type=str, required=True,
                        help='Path to the input video files.')
    args = parser.parse_args()
    
    file_list = args.video_paths
    
    reward_calculator = CameraAlignmentReward()

    # 创建模拟的地面真实 (GT) 外参数据
    num_frames = 24
    gt_extrinsics_list = []
    current_pose = np.eye(4)
    
    distance = 0.05  # 每帧沿z轴平移0.1个单位
    #gt_extrinsics_list.append(current_pose.copy())
    for i in range(num_frames):
        # 左前方运动
        yaw_per_frame = 0.05
        forward = 0.1
        
        pose = np.eye(4, dtype=np.float32)
        
        # 旋转矩阵（绕Y轴转向）
        cos_yaw = np.cos(yaw_per_frame)
        sin_yaw = np.sin(yaw_per_frame)
        pose[0, 0] = cos_yaw
        pose[0, 2] = sin_yaw
        pose[2, 0] = -sin_yaw
        pose[2, 2] = cos_yaw
        pose[2, 3] = -forward
        
        transform = torch.as_tensor(pose)
        current_pose = current_pose @ transform.numpy()
        gt_extrinsics_list.append(current_pose.copy())
    
    mock_gt_extrinsics = np.array(gt_extrinsics_list)
    
    for VIDEO_PATH in file_list:
        if not os.path.exists(VIDEO_PATH):
            print(f"错误: 示例视频文件未找到于 '{VIDEO_PATH}'")
        else:
            try:
                with imageio.get_reader(VIDEO_PATH) as reader:
                    video_frame_count = reader.count_frames()
                    # 读取 video，转成 tensor [B, F, H, W]，范围 [-1 1]
                    video_frames = []
                    for i, frame in enumerate(reader):
                        if i >= reader.count_frames():
                            break
                        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 * 2 - 1  # 归一化到 [-1, 1]
                        video_frames.append(frame_tensor)
                    video_tensor = torch.stack(video_frames, dim=0)  # [F, C, H, W]
                video_tensor = video_tensor.unsqueeze(0)  # [1, F, C, H, W]
                if (video_frame_count // 4) != num_frames:
                    print(f"警告: 视频帧数 ({video_frame_count // 4}) 与模拟GT帧数 ({num_frames}) 不符。正在调整GT数据以匹配。")
                    if video_frame_count < num_frames:
                        mock_gt_extrinsics = mock_gt_extrinsics[:video_frame_count]
                    else:
                        last_gt = mock_gt_extrinsics[-1]
                        padding = np.repeat(last_gt[np.newaxis, :, :], video_frame_count - num_frames, axis=0)
                        mock_gt_extrinsics = np.concatenate([mock_gt_extrinsics, padding], axis=0)
                # print(video_tensor.shape)
                reward_info = reward_calculator.calculate_reward(video_tensor, mock_gt_extrinsics)

                print("\n--- 对齐奖励结果 ---")
                print(f"平均旋转误差: {reward_info['mean_rotation_error_degrees']:.2f} 度")
                print(f"平均平移误差 (尺度对齐后): {reward_info['mean_translation_error']:.4f}")
                print(f"计算出的轨迹尺度因子: {reward_info['translation_scale_factor']:.4f}")
                print("-" * 20)
                print(f"旋转奖励: {reward_info['rotation_reward']:.4f}")
                print(f"平移奖励: {reward_info['translation_reward']:.4f}")
                print(f"最终加权总奖励: {reward_info['total_reward']:.4f}")
                print("----------------------\n")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"在计算奖励时发生错误: {e}")