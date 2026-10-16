"""
Trích xuất đặc trưng hình học cơ thể từ MediaPipe Pose Landmarks.
Xử lý độ tin cậy visibility và chuyển đổi thành FrameRecord cho bộ đệm thời gian.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any
import numpy as np


@dataclass
class FrameRecord:
    timestamp_sec: float
    frame_index: int
    pose_valid: bool
    keypoints: Dict[int, Tuple[float, float, float]] = field(default_factory=dict) # id -> (x, y, vis)
    shoulder_midpoint: Optional[Tuple[float, float]] = None # (x, y)
    hip_midpoint: Optional[Tuple[float, float]] = None      # (x, y)
    bbox: Optional[Tuple[float, float, float, float]] = None # (xmin, ymin, xmax, ymax) chuẩn hóa [0, 1]
    box_w: float = 0.0
    box_h: float = 0.0
    bbox_aspect_ratio: float = 0.0
    torso_angle_deg: float = 0.0
    fall_probability: Optional[float] = None
    cnn_timestamp: Optional[float] = None


def extract_pose_features(
    landmarks,
    timestamp_sec: float,
    frame_index: int = 0,
    min_visibility: float = 0.5,
    fall_prob: Optional[float] = None,
    cnn_timestamp: Optional[float] = None,
) -> FrameRecord:
    """
    Trích xuất đặc trưng tư thế từ danh sách landmarks của MediaPipe.
    Quy tắc an toàn: Không thay thế tọa độ thiếu bằng số 0; nếu vai hoặc hông
    không đủ visibility, đánh dấu pose_valid=False.
    """
    if landmarks is None or len(landmarks) < 25:
        return FrameRecord(
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            pose_valid=False,
            fall_probability=fall_prob,
            cnn_timestamp=cnn_timestamp,
        )

    # MediaPipe indices:
    # 11: LEFT_SHOULDER, 12: RIGHT_SHOULDER
    # 23: LEFT_HIP, 24: RIGHT_HIP
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    l_hip = landmarks[23]
    r_hip = landmarks[24]

    kp_dict = {}
    for idx, lm in enumerate(landmarks):
        kp_dict[idx] = (lm.x, lm.y, getattr(lm, "visibility", 1.0))

    # Kiểm tra chất lượng vai và hông
    sh_vis = (kp_dict[11][2] + kp_dict[12][2]) / 2.0
    hip_vis = (kp_dict[23][2] + kp_dict[24][2]) / 2.0

    if sh_vis < min_visibility or hip_vis < min_visibility:
        return FrameRecord(
            timestamp_sec=timestamp_sec,
            frame_index=frame_index,
            pose_valid=False,
            keypoints=kp_dict,
            fall_probability=fall_prob,
            cnn_timestamp=cnn_timestamp,
        )

    shoulder_mid = (
        (l_sh.x + r_sh.x) / 2.0,
        (l_sh.y + r_sh.y) / 2.0,
    )
    hip_mid = (
        (l_hip.x + r_hip.x) / 2.0,
        (l_hip.y + r_hip.y) / 2.0,
    )

    # Tính Bounding Box từ các điểm có visibility >= min_visibility
    valid_points = [
        (lm.x, lm.y) for lm in landmarks if getattr(lm, "visibility", 1.0) >= min_visibility
    ]
    if not valid_points:
        valid_points = [(shoulder_mid[0], shoulder_mid[1]), (hip_mid[0], hip_mid[1])]

    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    xmin, xmax = max(0.0, min(xs)), min(1.0, max(xs))
    ymin, ymax = max(0.0, min(ys)), min(1.0, max(ys))

    box_w = max(0.01, xmax - xmin)
    box_h = max(0.01, ymax - ymin)
    aspect_ratio = float(box_w / (box_h + 1e-5))

    # Góc thân người so với phương thẳng đứng
    dx = abs(shoulder_mid[0] - hip_mid[0])
    dy = abs(hip_mid[1] - shoulder_mid[1])
    torso_angle = float(np.degrees(np.arctan2(dx, dy + 1e-5)))

    return FrameRecord(
        timestamp_sec=timestamp_sec,
        frame_index=frame_index,
        pose_valid=True,
        keypoints=kp_dict,
        shoulder_midpoint=shoulder_mid,
        hip_midpoint=hip_mid,
        bbox=(xmin, ymin, xmax, ymax),
        box_w=box_w,
        box_h=box_h,
        bbox_aspect_ratio=aspect_ratio,
        torso_angle_deg=torso_angle,
        fall_probability=fall_prob,
        cnn_timestamp=cnn_timestamp,
    )

