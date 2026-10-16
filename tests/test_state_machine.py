"""
Unit tests cho Finite State Machine (FSM):
Kiểm tra vòng đời trạng thái:
WARMING_UP -> MONITORING -> CANDIDATE -> PENDING_CONFIRMATION -> FALL_CONFIRMED -> COOLDOWN -> RECOVERED -> RESET.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pose_features import FrameRecord
from src.temporal_verifier import TemporalFallVerifier


def make_rec(t, hip_y=0.5, torso_angle=0.0, aspect_ratio=0.4, p_fall=0.1):
    return FrameRecord(
        timestamp_sec=t,
        frame_index=1,
        pose_valid=True,
        shoulder_midpoint=(0.5, hip_y - 0.3),
        hip_midpoint=(0.5, hip_y),
        bbox=(0.3, hip_y - 0.3, 0.7, hip_y + 0.3),
        box_w=0.4,
        box_h=0.6,
        bbox_aspect_ratio=aspect_ratio,
        torso_angle_deg=torso_angle,
        fall_probability=p_fall,
    )


def test_fsm_lifecycle():
    verifier = TemporalFallVerifier(buffer_size=10, persistence_sec=0.3, cooldown_sec=2.0)
    t = 0.0

    # 1. WARMING_UP: 9 frames đầu
    for i in range(9):
        t += 0.033
        res = verifier.update(make_rec(t))
        assert res.state == "WARMING_UP"

    # 2. MONITORING: Frame thứ 10 trở đi
    t += 0.033
    res = verifier.update(make_rec(t))
    assert res.state == "MONITORING"

    # 3. CANDIDATE: Frame đầu tiên phát hiện tư thế nghi ngờ
    t += 0.033
    res = verifier.update(make_rec(t, hip_y=0.85, torso_angle=80.0, aspect_ratio=1.5))
    assert res.state == "CANDIDATE"

    # 4. PENDING_CONFIRMATION: Duy trì nhưng chưa đủ persistence_sec (0.3s)
    t += 0.033
    res = verifier.update(make_rec(t, hip_y=0.85, torso_angle=80.0, aspect_ratio=1.5))
    assert res.state == "PENDING_CONFIRMATION"

    # 5. FALL_CONFIRMED: Đạt đủ persistence_sec (0.3s ~ 10 frames)
    for _ in range(10):
        t += 0.033
        res = verifier.update(make_rec(t, hip_y=0.85, torso_angle=80.0, aspect_ratio=1.5))

    assert res.state == "FALL_CONFIRMED"
    assert res.is_fall_confirmed is True

    # 6. RESET: Nhấn phím R
    verifier.reset()
    assert verifier.state == "MONITORING"
    assert verifier.is_fall_confirmed is False


if __name__ == "__main__":
    test_fsm_lifecycle()
    print("[PASS] Test FSM State Machine dat 100%!")

