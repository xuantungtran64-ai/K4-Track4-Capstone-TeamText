"""
Chốt 2: Bộ xác minh thời gian (Temporal Verifier) và Máy trạng thái hữu hạn (FSM).
Sử dụng bộ đệm trượt 10+ frames, tính toán vận tốc rơi dọc theo delta_t thực tế,
phân biệt cúi người/ngồi với ngã thật bằng kiểm tra tư thế nằm duy trì (persistence).
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from src.pose_features import FrameRecord


@dataclass
class DetectionResult:
    state: str                          # WARMING_UP, MONITORING, CANDIDATE, PENDING_CONFIRMATION, FALL_CONFIRMED, COOLDOWN, UNCERTAIN
    is_fall_confirmed: bool             # Cờ kích hoạt cảnh báo khẩn cấp
    fall_probability: float             # Xác suất ngã từ CNN (hoặc 0.0 nếu chưa có)
    vy: float                           # Vận tốc rơi dọc hông (chiều cao màn hình / giây)
    torso_angle: float                  # Góc nghiêng thân người (độ)
    aspect_ratio: float                 # Tỷ lệ khung bao W / H
    reason_codes: List[str]             # Danh sách lý do quyết định
    features_debug: Dict[str, Any]      # Chi tiết kỹ thuật phục vụ debug/tuning
    elapsed_fall_time: float = 0.0      # Thời gian duy trì tư thế ngã (giây)


class TemporalFallVerifier:
    """
    Bộ xác minh thời gian rule-based kết hợp CNN.
    Quy tắc quan trọng: CNN là nguồn bằng chứng, không phải cổng chặn tuyệt đối.
    Nếu CNN không báo nhưng chuyển động hình học bất thường, vẫn kích hoạt xác minh.
    """

    def __init__(
        self,
        buffer_size: int = 10,
        cnn_threshold: float = 0.65,
        bbox_r_min: float = 1.05,
        torso_angle_min_deg: float = 55.0,
        rapid_drop_vy: float = 0.45,
        persistence_sec: float = 0.8,
        cooldown_sec: float = 5.0,
        min_visibility: float = 0.5,
    ):
        self.buffer_size = buffer_size
        self.cnn_threshold = cnn_threshold
        self.bbox_r_min = bbox_r_min
        self.torso_angle_min_deg = torso_angle_min_deg
        self.rapid_drop_vy = rapid_drop_vy
        self.persistence_sec = persistence_sec
        self.cooldown_sec = cooldown_sec
        self.min_visibility = min_visibility

        self.buffer = deque(maxlen=self.buffer_size)
        self.state = "WARMING_UP"
        self.candidate_start_time: Optional[float] = None
        self.last_confirmed_time: Optional[float] = None
        self.is_fall_confirmed = False
        self.had_motion_trigger = False

        # Baseline chiều cao khi người đang đứng thẳng để chuẩn hóa độ rơi
        self.standing_height_baseline: Optional[float] = None

    def reset(self):
        """Khôi phục trạng thái về ban đầu (khi y tá nhấn [R])."""
        self.buffer.clear()
        self.state = "MONITORING"
        self.candidate_start_time = None
        self.last_confirmed_time = None
        self.is_fall_confirmed = False
        self.had_motion_trigger = False

    def update(self, record: FrameRecord) -> DetectionResult:
        """
        Tiếp nhận một FrameRecord mới và cập nhật trạng thái FSM.
        """
        now = record.timestamp_sec
        reason_codes: List[str] = []
        features_debug: Dict[str, Any] = {}

        # 1. Nếu đang ở trạng thái đã xác nhận ngã (FALL_CONFIRMED)
        if self.is_fall_confirmed:
            elapsed_since_confirm = now - (self.last_confirmed_time or now)
            # Kiểm tra xem người bệnh đã đứng dậy an toàn chưa
            if record.pose_valid and record.torso_angle_deg < 45.0 and record.bbox_aspect_ratio < 0.9:
                # Đã phục hồi tư thế đứng thẳng
                self.is_fall_confirmed = False
                self.last_confirmed_time = None
                self.candidate_start_time = None
                self.had_motion_trigger = False
                self.state = "MONITORING"
                reason_codes.append("RECOVERED_STANDING")
            else:
                self.state = "FALL_CONFIRMED"
                if elapsed_since_confirm < self.cooldown_sec:
                    reason_codes.append("COOLDOWN_ACTIVE")
                else:
                    reason_codes.append("STILL_DOWN_AFTER_COOLDOWN")

                elapsed_fall = (now - self.candidate_start_time) if self.candidate_start_time else elapsed_since_confirm
                return DetectionResult(
                    state=self.state,
                    is_fall_confirmed=True,
                    fall_probability=record.fall_probability or 0.0,
                    vy=0.0,
                    torso_angle=record.torso_angle_deg,
                    aspect_ratio=record.bbox_aspect_ratio,
                    reason_codes=reason_codes,
                    features_debug={"elapsed_since_confirm": elapsed_since_confirm},
                    elapsed_fall_time=elapsed_fall,
                )

        # 2. Xử lý khi mất mốc cơ thể hoặc độ tin cậy thấp
        if not record.pose_valid or record.hip_midpoint is None or record.shoulder_midpoint is None:
            self.buffer.append(record)
            reason_codes.append("LOW_VISIBILITY_OR_NO_POSE")
            self.state = "UNCERTAIN"
            return DetectionResult(
                state=self.state,
                is_fall_confirmed=False,
                fall_probability=record.fall_probability or 0.0,
                vy=0.0,
                torso_angle=0.0,
                aspect_ratio=0.0,
                reason_codes=reason_codes,
                features_debug={"pose_valid": False},
                elapsed_fall_time=0.0,
            )

        # Lưu record hợp lệ vào bộ đệm
        self.buffer.append(record)

        # 3. Kiểm tra giai đoạn Warming up (chưa đủ số frame tối thiểu)
        valid_records = [r for r in self.buffer if r.pose_valid and r.hip_midpoint is not None]
        if len(valid_records) < self.buffer_size:
            self.state = "WARMING_UP"
            reason_codes.append("BUFFER_WARMING_UP")
            return DetectionResult(
                state=self.state,
                is_fall_confirmed=False,
                fall_probability=record.fall_probability or 0.0,
                vy=0.0,
                torso_angle=record.torso_angle_deg,
                aspect_ratio=record.bbox_aspect_ratio,
                reason_codes=reason_codes,
                features_debug={"buffer_len": len(valid_records)},
                elapsed_fall_time=0.0,
            )

        # 4. Tính toán các đặc trưng chuyển động theo cửa sổ thời gian
        earliest_rec = valid_records[0]
        latest_rec = valid_records[-1]

        delta_t = latest_rec.timestamp_sec - earliest_rec.timestamp_sec
        # Trục Y của ảnh hướng xuống dưới: ngã xuống tức delta_y > 0
        delta_y_hip = latest_rec.hip_midpoint[1] - earliest_rec.hip_midpoint[1]
        vertical_speed = delta_y_hip / (delta_t + 1e-5) if delta_t > 0 else 0.0

        # Cập nhật baseline chiều cao khi đứng thẳng (torso_angle < 20, aspect_ratio < 0.6)
        if record.torso_angle_deg < 20.0 and record.bbox_aspect_ratio < 0.6:
            self.standing_height_baseline = record.box_h

        hip_drop_ratio = 0.0
        if self.standing_height_baseline and self.standing_height_baseline > 0.05:
            hip_drop_ratio = delta_y_hip / self.standing_height_baseline

        features_debug = {
            "delta_t": delta_t,
            "delta_y_hip": delta_y_hip,
            "vertical_speed": vertical_speed,
            "hip_drop_ratio": hip_drop_ratio,
            "torso_angle": record.torso_angle_deg,
            "aspect_ratio": record.bbox_aspect_ratio,
            "fall_probability": record.fall_probability,
        }

        # 5. Đánh giá dấu hiệu ngã và phân biệt cúi người (Bending) với nằm sàn (Lying)
        has_fast_drop = (vertical_speed > self.rapid_drop_vy) or (hip_drop_ratio > 0.40)
        if has_fast_drop:
            self.had_motion_trigger = True
            reason_codes.append("FAST_DOWNWARD_MOTION")

        # Tư thế nằm sàn thật sự:
        # - Khung bao nằm ngang W/H > BBOX_R_MIN
        # - Hoặc vai/đầu tụt sát đáy khung hình (y > 0.80)
        # - Hoặc cả vai và hông đều hạ thấp sát sàn (hip_y > 0.75 và torso_angle > 50)
        is_true_lying_posture = (
            record.bbox_aspect_ratio > self.bbox_r_min
            or record.shoulder_midpoint[1] > 0.80
            or (record.hip_midpoint[1] > 0.75 and record.torso_angle_deg > 50.0)
        )

        # Dấu hiệu nghi ngờ mất thăng bằng hoặc cúi người
        is_bent_or_unbalanced = (
            record.torso_angle_deg > self.torso_angle_min_deg
            or record.bbox_aspect_ratio > 1.0
            or record.shoulder_midpoint[1] > 0.75
        )

        has_cnn_candidate = (
            record.fall_probability is not None
            and record.fall_probability >= self.cnn_threshold
        )
        if has_cnn_candidate:
            reason_codes.append("CNN_SUSPECTED")

        if is_true_lying_posture:
            reason_codes.append("HORIZONTAL_POSTURE")
        elif is_bent_or_unbalanced:
            reason_codes.append("POSTURE_UNBALANCED")

        # 6. Trigger Candidate FSM
        is_candidate_triggered = has_fast_drop or is_bent_or_unbalanced or is_true_lying_posture or has_cnn_candidate

        elapsed_fall_time = 0.0
        if is_candidate_triggered:
            if self.candidate_start_time is None:
                self.candidate_start_time = now
                self.state = "CANDIDATE"
                reason_codes.append("CANDIDATE_TRIGGERED")
            else:
                self.state = "PENDING_CONFIRMATION"
                reason_codes.append("CHECKING_POSTURE_PERSISTENCE")

            elapsed_fall_time = now - self.candidate_start_time

            # Chốt 2 - Xác minh Té Ngã Khẳng Định:
            # - Duy trì liên tục >= persistence_sec
            # - VÀ tại thời điểm xác nhận người phải ở tư thế nằm sàn thật sự (is_true_lying_posture)
            #   hoặc có chuyển động rơi đột ngột kết hợp nằm ngang.
            if elapsed_fall_time >= self.persistence_sec and is_true_lying_posture:
                self.is_fall_confirmed = True
                self.state = "FALL_CONFIRMED"
                self.last_confirmed_time = now
                reason_codes.append("POSTURE_PERSISTED_FALL_CONFIRMED")
        else:
            # Người đứng thẳng dậy bình thường
            if record.torso_angle_deg < 45.0 and record.bbox_aspect_ratio < 0.9:
                self.state = "MONITORING"
                self.candidate_start_time = None
                self.had_motion_trigger = False
                reason_codes.append("NORMAL_MONITORING")
            else:
                self.state = "MONITORING"

        return DetectionResult(
            state=self.state,
            is_fall_confirmed=self.is_fall_confirmed,
            fall_probability=record.fall_probability or 0.0,
            vy=vertical_speed,
            torso_angle=record.torso_angle_deg,
            aspect_ratio=record.bbox_aspect_ratio,
            reason_codes=reason_codes,
            features_debug=features_debug,
            elapsed_fall_time=elapsed_fall_time,
        )

