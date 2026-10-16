"""
Hệ thống Pipeline phát hiện ngã 2 chốt hoàn chỉnh (Two-Stage Fall Detection Pipeline).
Kết nối Capture -> MediaPipe Pose Lite -> CNN Classifier -> Ring Buffer -> Temporal Verifier -> CCTV Overlay.
"""

import time
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
import mediapipe as mp

import config
from src.pose_features import extract_pose_features, FrameRecord
from src.fall_image_classifier import FallImageClassifier
from src.temporal_verifier import TemporalFallVerifier, DetectionResult
from src.overlay import render_hospital_overlay


class FallDetectionPipeline:
    """
    Pipeline tích hợp xử lý real-time trên CPU:
    - Chốt 1: CNN phân loại ảnh (TFLite / LiteRT) chạy giới hạn CNN_MAX_FPS.
    - Chốt 2: MediaPipe Pose + Bộ đệm thời gian 10+ frame xác minh chuyển động và tư thế.
    """

    def __init__(
        self,
        model_complexity: int = config.MODEL_COMPLEXITY,
        model_path: str = config.MODEL_PATH,
        cnn_max_fps: float = config.CNN_MAX_FPS,
        cnn_threshold: float = config.CNN_THRESHOLD,
        cnn_input_mode: str = config.CNN_INPUT_MODE,
        buffer_size: int = config.BUFFER_SIZE,
        bbox_r_min: float = config.BBOX_R_MIN,
        torso_angle_min_deg: float = config.TORSO_ANGLE_MIN_DEG,
        rapid_drop_vy: float = config.RAPID_DROP_VY,
        persistence_sec: float = config.PERSISTENCE_SEC,
        cooldown_sec: float = config.COOLDOWN_SEC,
        min_visibility: float = config.MIN_VISIBILITY,
        inference_threads: int = config.INFERENCE_THREADS,
    ):
        # 1. MediaPipe Pose Lite
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            model_complexity=model_complexity,
            smooth_landmarks=config.SMOOTH_LANDMARKS,
        )

        # 2. Chốt 1: CNN Classifier
        self.classifier = FallImageClassifier(
            model_path=model_path,
            num_threads=inference_threads,
            max_fps=cnn_max_fps,
            input_mode=cnn_input_mode,
            crop_margin=config.CROP_MARGIN,
        )

        # 3. Chốt 2: Temporal Verifier & FSM
        self.verifier = TemporalFallVerifier(
            buffer_size=buffer_size,
            cnn_threshold=cnn_threshold,
            bbox_r_min=bbox_r_min,
            torso_angle_min_deg=torso_angle_min_deg,
            rapid_drop_vy=rapid_drop_vy,
            persistence_sec=persistence_sec,
            cooldown_sec=cooldown_sec,
            min_visibility=min_visibility,
        )

        self.frame_index = 0
        self.latest_result: Optional[DetectionResult] = None

    def reset_alarm(self):
        """Khôi phục trạng thái hệ thống khi người bệnh đã an toàn."""
        self.verifier.reset()

    def process(
        self,
        frame: np.ndarray,
        current_time: Optional[float] = None,
        fps_display: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Xử lý 1 frame hình ảnh và trả về frame có overlay cùng telemetry dictionary.
        """
        if current_time is None:
            current_time = time.time()

        self.frame_index += 1
        h, w = frame.shape[:2]

        # Tối ưu CPU Intel Core i5: Resize nhẹ cho bước suy luận Pose (MediaPipe dùng tensor 256x256)
        if w > config.SCALE_INFER_WIDTH:
            scale_ratio = float(config.SCALE_INFER_WIDTH) / w
            infer_h = max(16, int(h * scale_ratio))
            infer_frame = cv2.resize(frame, (config.SCALE_INFER_WIDTH, infer_h), interpolation=cv2.INTER_LINEAR)
        else:
            infer_frame = frame

        frame_rgb = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        pose_results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        landmarks = None
        if pose_results and pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

        # Tính toán đặc trưng hình học
        record = extract_pose_features(
            landmarks=landmarks,
            timestamp_sec=current_time,
            frame_index=self.frame_index,
            min_visibility=config.MIN_VISIBILITY,
        )

        # Chốt 1: Chạy CNN phân loại ảnh (được giới hạn tần suất bởi CNN_MAX_FPS)
        bbox = record.bbox if (record.pose_valid and record.bbox) else None
        if config.CNN_INPUT_MODE == "landmark_crop" and not record.pose_valid:
            p_fall = self.classifier.last_fall_prob
        else:
            p_fall = self.classifier.predict_fall_probability(
                frame=frame,
                bbox=bbox,
                current_time=current_time,
            )
        record.fall_probability = p_fall
        record.cnn_timestamp = self.classifier.last_infer_time

        # Chốt 2: Cập nhật Temporal Verifier
        detection_result = self.verifier.update(record)
        self.latest_result = detection_result

        # Vẽ giao diện giám sát bệnh viện
        annotated_frame = render_hospital_overlay(
            frame=frame,
            result=detection_result,
            record=record,
            fps=fps_display,
            cnn_latency_ms=self.classifier.last_latency_ms,
        )

        telemetry = {
            "status": detection_result.state,
            "status_text": f"[{detection_result.state}] {', '.join(detection_result.reason_codes[-2:]) if detection_result.reason_codes else 'NORMAL'}",
            "vy": detection_result.vy,
            "torso_angle": detection_result.torso_angle,
            "aspect_ratio": detection_result.aspect_ratio,
            "is_fall_confirmed": detection_result.is_fall_confirmed,
            "elapsed_fall_time": detection_result.elapsed_fall_time,
            "has_detection": record.pose_valid,
            "fall_probability": detection_result.fall_probability,
            "reason_codes": detection_result.reason_codes,
            "cnn_latency_ms": self.classifier.last_latency_ms,
        }

        return annotated_frame, telemetry

    def close(self):
        """Giải phóng tài nguyên."""
        self.pose.close()
