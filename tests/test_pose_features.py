"""
Unit tests cho module trích xuất đặc trưng hình học cơ thể (src/pose_features.py).
Kiểm tra tính toán mốc tọa độ, góc nghiêng thân, tỷ lệ khung bao và xử lý visibility.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pose_features import extract_pose_features, FrameRecord


class MockLandmark:
    def __init__(self, x, y, visibility=0.9):
        self.x = x
        self.y = y
        self.visibility = visibility


def create_33_landmarks(sh_left=(0.45, 0.3), sh_right=(0.55, 0.3),
                        hip_left=(0.45, 0.6), hip_right=(0.55, 0.6),
                        vis=0.9):
    lms = [MockLandmark(0.5, 0.5, vis) for _ in range(33)]
    lms[11] = MockLandmark(sh_left[0], sh_left[1], vis)
    lms[12] = MockLandmark(sh_right[0], sh_right[1], vis)
    lms[23] = MockLandmark(hip_left[0], hip_left[1], vis)
    lms[24] = MockLandmark(hip_right[0], hip_right[1], vis)
    return lms


def test_standing_upright_features():
    # Đứng thẳng: vai y=0.3, hông y=0.6 -> torso_angle ~ 0 deg
    lms = create_33_landmarks(
        sh_left=(0.45, 0.3), sh_right=(0.55, 0.3),
        hip_left=(0.45, 0.6), hip_right=(0.55, 0.6),
    )
    record = extract_pose_features(lms, timestamp_sec=10.0, frame_index=1)

    assert record.pose_valid is True
    assert record.shoulder_midpoint == (0.5, 0.3)
    assert record.hip_midpoint == (0.5, 0.6)
    assert record.torso_angle_deg < 5.0, f"Góc thân đứng thẳng phải gần 0: {record.torso_angle_deg}"


def test_lying_horizontal_features():
    # Nằm ngang: vai x=0.2, y=0.85; hông x=0.5, y=0.85 -> torso_angle ~ 90 deg
    lms = create_33_landmarks(
        sh_left=(0.20, 0.85), sh_right=(0.22, 0.85),
        hip_left=(0.50, 0.85), hip_right=(0.52, 0.85),
    )
    record = extract_pose_features(lms, timestamp_sec=10.0, frame_index=1)

    assert record.pose_valid is True
    assert record.torso_angle_deg > 85.0, f"Góc thân nằm ngang phải gần 90: {record.torso_angle_deg}"


def test_low_visibility_treated_as_invalid():
    # Vai hoặc hông có visibility < 0.5 phải được đánh dấu pose_valid=False (không thay bằng 0)
    lms = create_33_landmarks(vis=0.2)
    record = extract_pose_features(lms, timestamp_sec=10.0, frame_index=1, min_visibility=0.5)

    assert record.pose_valid is False
    assert record.shoulder_midpoint is None
    assert record.hip_midpoint is None


def test_none_landmarks_handled_gracefully():
    record = extract_pose_features(None, timestamp_sec=10.0, frame_index=1)
    assert record.pose_valid is False


if __name__ == "__main__":
    test_standing_upright_features()
    test_lying_horizontal_features()
    test_low_visibility_treated_as_invalid()
    test_none_landmarks_handled_gracefully()
    print("[PASS] Tat ca unit tests cho Pose Features deu dat 100%!")

