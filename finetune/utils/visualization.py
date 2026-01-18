import cv2
from PIL import Image
import torch

def extract_target_poses_to_txt(tensor, output_file="target_poses.txt"):
    """
    从N×13的tensor中提取target位姿并导出到txt文件
    
    参数:
    tensor: 输入tensor，形状为(N, 13)
            - 每行前12列: 3×4相机位姿矩阵的展平结果
            - 每行最后一列: 1表示condition, -1表示target
    output_file: 输出文件名
    """
    
    # 确保输入是tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    
    # 提取target行 (最后一列为0的行)
    target_mask = tensor[:, -1] == 0
    target_data = tensor[target_mask]
    
    # 准备输出文件
    with open(output_file, 'w') as f:
        for i, row in enumerate(target_data):
            # 提取前12个元素作为3×4矩阵
            pose_3x4 = row[:12].reshape(3, 4)
            
            # 转换为4×4齐次坐标矩阵
            pose_4x4 = torch.eye(4)
            pose_4x4[:3, :] = pose_3x4
            
            # 写入文件
            f.write(f"Target Pose {i+1}:\n")
            
            # 写入4×4矩阵，每行4个元素
            for j in range(4):
                line = " ".join([f"{pose_4x4[j, k]:.6f}" for k in range(4)])
                f.write(line + "\n")
            
            f.write("\n")  # 空行分隔不同的位姿
    
    print(f"target位姿已成功导出到: {output_file}")
