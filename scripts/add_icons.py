import numpy as np

def overlay_controls(frame_img, pose_vec, icons):
    """
    Overlay control icons (WASD and arrows) on frame based on camera pose
    pose_vec: 12 elements (flattened 3x4 matrix) + mask
    """
    if pose_vec is None or np.all(pose_vec[:12] == 0):
        return frame_img
        
    # Extract translation vector (based on flattened 3x4 matrix indices)
    # [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz]
    tx = pose_vec[3]
    # ty = pose_vec[7]
    tz = pose_vec[11]
    
    # Extract rotation (yaw and pitch)
    # Yaw: around Y axis. sin(yaw) = r02, cos(yaw) = r00
    r00 = pose_vec[0]
    r02 = pose_vec[2]
    yaw = np.arctan2(r02, r00)
    
    # Pitch: around X axis. sin(pitch) = -r12, cos(pitch) = r22
    r12 = pose_vec[6]
    r22 = pose_vec[10]
    pitch = np.arctan2(-r12, r22)
    
    # Threshold for key activation
    TRANS_THRESH = 0.01
    ROT_THRESH = 0.005
    
    # Determine key states
    # Translation (WASD)
    # Assume -Z is forward, +X is right
    is_forward = tz < -TRANS_THRESH
    is_backward = tz > TRANS_THRESH
    is_left = tx < -TRANS_THRESH
    is_right = tx > TRANS_THRESH
    
    # Rotation (arrows)
    # Yaw: + is left, - is right
    is_turn_left = yaw > ROT_THRESH
    is_turn_right = yaw < -ROT_THRESH
    
    # Pitch: + is down, - is up
    is_turn_up = pitch < -ROT_THRESH
    is_turn_down = pitch > ROT_THRESH
    
    W, H = frame_img.size
    spacing = 60
    
    def paste_icon(name_active, name_inactive, is_active, x, y):
        name = name_active if is_active else name_inactive
        if name in icons:
            icon = icons[name]
        # Paste using alpha channel
            frame_img.paste(icon, (int(x), int(y)), icon)
    
    # Overlay WASD (bottom left)
    base_x_right = 100
    base_y = H - 100
    
    # W
    paste_icon('move_forward.png', 'not_move_forward.png', is_forward, base_x_right, base_y - spacing)
    # A
    paste_icon('move_left.png', 'not_move_left.png', is_left, base_x_right - spacing, base_y)
    # S
    paste_icon('move_backward.png', 'not_move_backward.png', is_backward, base_x_right, base_y)
    # D
    paste_icon('move_right.png', 'not_move_right.png', is_right, base_x_right + spacing, base_y)
    
    # Overlay arrows (bottom right)
    base_x_left = W - 150
    
    # ↑
    paste_icon('turn_up.png', 'not_turn_up.png', is_turn_up, base_x_left, base_y - spacing)
    # ←
    paste_icon('turn_left.png', 'not_turn_left.png', is_turn_left, base_x_left - spacing, base_y)
    # ↓
    paste_icon('turn_down.png', 'not_turn_down.png', is_turn_down, base_x_left, base_y)
    # →
    paste_icon('turn_right.png', 'not_turn_right.png', is_turn_right, base_x_left + spacing, base_y)
    
    return frame_img