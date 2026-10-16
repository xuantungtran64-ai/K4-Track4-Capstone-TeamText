"""
Unit tests cho Temporal Fall Verifier theo đúng 8 kịch bản bắt buộc của Plan A Updated:
1. Đứng yên 10 frame -> không có candidate Fall.
2. Ngồi xuống chậm -> không xác nhận Fall chỉ vì hông hạ thấp.
3. Cúi nhặt đồ -> không xác nhận nếu không chuyển sang tư thế nằm duy trì.
4. Ngã mô phỏng: hông đi xuống nhanh, thân chuyển ngang, giữ tư thế -> FALL_CONFIRMED.
5. Nằm sẵn từ đầu video -> CNN score cao nhưng thiếu chuyển động trước đó -> CANDIDATE/PENDING, không báo ngã đột ngột.
6. Landmarks mất/visibility thấp -> UNCERTAIN, không dùng số 0 gây báo động giả.
7. Timestamp không đều -> vận tốc tính bằng delta_t thực, không theo FPS cố định.
8. Một sự kiện kéo dài nhiều frame -> chỉ phát một cảnh báo trong cooldown.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.pose_features import FrameRecord
from src.temporal_verifier import TemporalFallVerifier


def make_record(
    timestamp: float,
    hip_y: float = 0.6,
    shoulder_y: float = 0.3,
    hip_x: float = 0.5,
    shoulder_x: float = 0.5,
    aspect_ratio: float = 0.4,
    torso_angle: float = 0.0,
    fall_prob: float = 0.1,
    pose_valid: bool = True,
    box_h: float = 0.6,
    box_w: float = 0.25,
) -> FrameRecord:
    return FrameRecord(
        timestamp_sec=timestamp,
        frame_index=1,
        pose_valid=pose_valid,
        shoulder_midpoint=(shoulder_x, shoulder_y) if pose_valid else None,
        hip_midpoint=(hip_x, hip_y) if pose_valid else None,
        bbox=(0.3, shoulder_y, 0.7, hip_y + 0.3) if pose_valid else None,
        box_w=box_w,
        box_h=box_h,
        bbox_aspect_ratio=aspect_ratio,
        torso_angle_deg=torso_angle,
        fall_probability=fall_prob,
    )


# TEST 1: Đứng yên 10 frame -> không có candidate Fall (MONITORING)
def test_1_standing_still():
    verifier = TemporalFallVerifier(buffer_size=10)
    for i in range(12):
        rec = make_record(timestamp=100.0 + i * 0.033, torso_angle=5.0, aspect_ratio=0.35, hip_y=0.6)
        res = verifier.update(rec)

    assert res.state == "MONITORING", f"Test 1 Failed: {res.state} != MONITORING"
    assert not res.is_fall_confirmed


# TEST 2: Ngồi xuống chậm -> không xác nhận Fall chỉ vì hông hạ thấp
def test_2_slow_sitting():
    verifier = TemporalFallVerifier(buffer_size=10)
    # Hông hạ từ 0.6 xuống 0.85 chậm rãi trong 3 giây (90 frames)
    t = 100.0
    for i in range(90):
        t += 0.033
        curr_hip_y = 0.6 + (0.85 - 0.6) * (i / 90.0)
        rec = make_record(timestamp=t, hip_y=curr_hip_y, torso_angle=10.0, aspect_ratio=0.5)
        res = verifier.update(rec)

    assert not res.is_fall_confirmed, "Test 2 Failed: Ngoi cham bi nhan nham thanh Fall Confirmed!"
    assert res.state in ("MONITORING", "WARMING_UP")


# TEST 3: Cúi nhặt đồ -> không xác nhận nếu không chuyển sang tư thế nằm duy trì
def test_3_bending_pick_up():
    verifier = TemporalFallVerifier(buffer_size=10, persistence_sec=0.8)
    t = 100.0
    # Đứng thẳng 10 frames
    for i in range(10):
        t += 0.033
        verifier.update(make_record(timestamp=t, torso_angle=5.0, aspect_ratio=0.4, hip_y=0.6))

    # Cúi nhặt đồ trong 0.5s (15 frames) -> torso_angle cao 75 deg nhưng aspect_ratio < 0.9 và hip_y chưa sát sàn
    for i in range(15):
        t += 0.033
        rec = make_record(timestamp=t, torso_angle=75.0, aspect_ratio=0.7, hip_y=0.6)
        res = verifier.update(rec)

    assert res.state in ("CANDIDATE", "PENDING_CONFIRMATION"), f"Phai la Canh bao nghi ngo: {res.state}"
    assert not res.is_fall_confirmed, "Test 3 Failed: Cui nhat do bi nhan nham thanh Fall Confirmed!"

    # Đứng thẳng dậy ngay sau đó
    for i in range(10):
        t += 0.033
        res = verifier.update(make_record(timestamp=t, torso_angle=5.0, aspect_ratio=0.4, hip_y=0.6))

    assert res.state == "MONITORING", f"Test 3 Recovery Failed: {res.state} != MONITORING"


# TEST 4: Ngã mô phỏng: hông đi xuống nhanh, thân chuyển ngang, giữ tư thế -> FALL_CONFIRMED
def test_4_simulated_fall():
    verifier = TemporalFallVerifier(buffer_size=10, persistence_sec=0.4)
    t = 100.0
    # Đứng 10 frames
    for _ in range(10):
        t += 0.033
        verifier.update(make_record(timestamp=t, hip_y=0.5, shoulder_y=0.2, torso_angle=0.0, aspect_ratio=0.35))

    # Rơi nhanh trong 0.2s (hông tụt từ 0.5 xuống 0.85 -> vy > 1.5)
    for i in range(6):
        t += 0.033
        y = 0.5 + (0.85 - 0.5) * (i / 6.0)
        verifier.update(make_record(timestamp=t, hip_y=y, shoulder_y=y - 0.1, torso_angle=40.0, aspect_ratio=0.8))

    # Nằm sàn duy trì trong 0.6s (> persistence_sec = 0.4s)
    for _ in range(20):
        t += 0.033
        res = verifier.update(make_record(timestamp=t, hip_y=0.87, shoulder_y=0.85, torso_angle=88.0, aspect_ratio=1.6))

    assert res.state == "FALL_CONFIRMED", f"Test 4 Failed: {res.state} != FALL_CONFIRMED"
    assert res.is_fall_confirmed is True


# TEST 5: Nằm sẵn từ đầu video -> CNN score cao nhưng thiếu chuyển động trước đó
def test_5_already_lying():
    verifier = TemporalFallVerifier(buffer_size=10)
    t = 100.0
    # Đã nằm ngay từ frame 0, không có chuyển động rơi trước đó
    for i in range(10):
        t += 0.033
        rec = make_record(timestamp=t, hip_y=0.87, shoulder_y=0.85, torso_angle=88.0, aspect_ratio=1.6, fall_prob=0.90)
        res = verifier.update(rec)

    # Frame thứ 10 là frame đầu tiên qua buffer -> Kích hoạt CANDIDATE/PENDING, chưa phải FALL_CONFIRMED ngay lập tức
    assert res.state in ("CANDIDATE", "PENDING_CONFIRMATION", "WARMING_UP")


# TEST 6: Landmarks mất / visibility thấp -> UNCERTAIN, không dùng tọa độ số 0 gây báo ngã giả
def test_6_low_visibility():
    verifier = TemporalFallVerifier(buffer_size=10)
    t = 100.0
    # Đưa các frame invalid vào
    for i in range(10):
        t += 0.033
        rec = make_record(timestamp=t, pose_valid=False)
        res = verifier.update(rec)

    assert res.state in ("UNCERTAIN", "WARMING_UP")
    assert not res.is_fall_confirmed


# TEST 7: Timestamp không đều -> vận tốc tính bằng delta_t thực tế
def test_7_irregular_timestamps():
    verifier = TemporalFallVerifier(buffer_size=10)
    # 10 frames với timestamp cách nhau biến thiên (lag mạng / drop frame)
    times = [0.0, 0.05, 0.12, 0.25, 0.30, 0.45, 0.60, 0.85, 1.20, 1.80]
    for idx, t in enumerate(times):
        # Hông rơi từ 0.5 xuống 0.8
        y = 0.5 + 0.3 * (idx / 9.0)
        rec = make_record(timestamp=t, hip_y=y)
        res = verifier.update(rec)

    # delta_y = 0.3, delta_t = 1.80s -> vy = 0.3 / 1.8 = 0.166 (không bị tính sai thành drop nhanh)
    assert abs(res.vy - (0.3 / 1.8)) < 0.02


# TEST 8: Một sự kiện kéo dài nhiều frame -> chỉ phát một cảnh báo trong cooldown
def test_8_cooldown_single_alarm():
    verifier = TemporalFallVerifier(buffer_size=10, persistence_sec=0.2, cooldown_sec=3.0)
    t = 100.0
    # Đứng
    for _ in range(10):
        t += 0.033
        verifier.update(make_record(timestamp=t, hip_y=0.5, torso_angle=0.0, aspect_ratio=0.4))

    # Ngã và nằm xuống
    for _ in range(15):
        t += 0.033
        res = verifier.update(make_record(timestamp=t, hip_y=0.88, torso_angle=85.0, aspect_ratio=1.5))

    assert res.is_fall_confirmed is True

    # Tiếp tục nằm trong thời gian Cooldown (2.0s < cooldown_sec=3.0s)
    cooldown_seen = False
    for _ in range(60): # ~2s
        t += 0.033
        res = verifier.update(make_record(timestamp=t, hip_y=0.88, torso_angle=85.0, aspect_ratio=1.5))
        if "COOLDOWN_ACTIVE" in res.reason_codes:
            cooldown_seen = True

    assert cooldown_seen is True, "Test 8 Failed: Cooldown khong duoc kich hoat de chan canh bao lap!"


if __name__ == "__main__":
    test_1_standing_still()
    test_2_slow_sitting()
    test_3_bending_pick_up()
    test_4_simulated_fall()
    test_5_already_lying()
    test_6_low_visibility()
    test_7_irregular_timestamps()
    test_8_cooldown_single_alarm()
    print("[PASS] Tat ca 8/8 kịch ban bat buoc cho Temporal Fall Verifier deu dat 100%!")

