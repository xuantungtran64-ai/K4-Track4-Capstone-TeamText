"""
VINMEC ELDERLY FALL DETECTION DEMO - OPTION 2B (PLAN A UPDATED)
==============================================================
Hệ thống Camera AI giám sát an toàn cho bệnh nhân cao tuổi tại phòng bệnh/hành lang VinMec.
Kiến trúc 2 Chốt (Two-Stage Pipeline):
- Chốt 1: CNN phân loại ảnh Fall / No Fall (TFLite / LiteRT pretrained) trên CPU.
- Chốt 2: MediaPipe Pose Lite + Bộ đệm thời gian 10+ frame xác minh chuyển động và tư thế.

Tối ưu phần cứng CPU Intel Core i5:
- Tốc độ suy luận mượt mà >= 30 FPS.
- CNN được điều tiết tần suất tối ưu (CNN_MAX_FPS = 8).
- Bộ lọc chống báo động giả khi cúi nhặt đồ hoặc ngồi xuống từ từ.
- Giao diện Telemetry đầy đủ, trục trọng tâm cơ thể và cảnh báo khẩn cấp chớp đỏ.
"""

import sys
import time
import os
import cv2
import numpy as np

import config
from src.fall_detection_pipeline import FallDetectionPipeline


class VinmecFallDetector:
    """
    Bộ phân tích tư thế và phát hiện ngã chuyên dụng cho môi trường bệnh viện VinMec.
    Kế thừa kiến trúc 2 chốt của Plan A Updated, đồng thời tương thích hoàn toàn
    với Launcher UI và bộ test tự động.
    """

    def __init__(self, model_complexity=config.MODEL_COMPLEXITY):
        self.pipeline = FallDetectionPipeline(model_complexity=model_complexity)
        # Expose pose để hỗ trợ mock landmarks trong unit tests
        self.pose = self.pipeline.pose

    @property
    def status(self):
        st = self.pipeline.verifier.state
        if st == "MONITORING":
            return "NORMAL"
        elif st in ("CANDIDATE", "PENDING_CONFIRMATION"):
            return "SUSPECTED"
        return st

    @status.setter
    def status(self, val):
        if val == "NORMAL":
            self.pipeline.verifier.state = "MONITORING"
        elif val == "SUSPECTED":
            self.pipeline.verifier.state = "CANDIDATE"
        else:
            self.pipeline.verifier.state = val

    @property
    def is_fall_confirmed(self):
        return self.pipeline.verifier.is_fall_confirmed

    @is_fall_confirmed.setter
    def is_fall_confirmed(self, val):
        self.pipeline.verifier.is_fall_confirmed = val

    @property
    def fall_start_time(self):
        return self.pipeline.verifier.candidate_start_time

    @fall_start_time.setter
    def fall_start_time(self, val):
        self.pipeline.verifier.candidate_start_time = val

    def reset_alarm(self):
        """Khôi phục hệ thống về trạng thái bình thường (khi y tá đã hỗ trợ xong)."""
        self.pipeline.reset_alarm()

    def process(self, frame, current_time=None, fps_display=0.0):
        """
        Xử lý 1 frame hình ảnh và cập nhật trạng thái phát hiện ngã.
        :param frame: Frame ảnh BGR từ OpenCV
        :param current_time: Timestamp hiện tại (None để lấy time.time())
        :param fps_display: Tốc độ FPS hiển thị
        :return: (annotated_frame, telemetry)
        """
        if current_time is None:
            current_time = time.time()

        annotated_frame, telemetry = self.pipeline.process(
            frame, current_time=current_time, fps_display=fps_display
        )

        # Chuẩn hóa tên trạng thái tương thích với giao diện và test suite
        raw_state = telemetry["status"]
        if raw_state == "MONITORING":
            telemetry["status"] = "NORMAL"
        elif raw_state in ("CANDIDATE", "PENDING_CONFIRMATION"):
            telemetry["status"] = "SUSPECTED"

        return annotated_frame, telemetry

    def close(self):
        """Giải phóng tài nguyên."""
        self.pipeline.close()


def run_vinmec_fall_detection(source=0):
    """
    Hàm thực thi chính hệ thống giám sát ngã VinMec.
    :param source: 0 (webcam mặc định) hoặc đường dẫn file video mẫu.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[LOI] Khong the mo nguon video: {source}")
        return

    # Xác định tốc độ phát tự nhiên nếu là file video
    is_video_file = isinstance(source, str)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if is_video_file and video_fps and video_fps > 0:
        playback_delay = max(1, int(1000.0 / video_fps))
    else:
        playback_delay = 1

    # Tối ưu kích thước capture cho webcam
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    # Khởi tạo detector
    detector = VinmecFallDetector(model_complexity=config.MODEL_COMPLEXITY)

    prev_time = time.time()
    fps_display = 0.0
    frame_count = 0

    print("=" * 70)
    print(" VINMEC SMART WARD - 2-STAGE FALL DETECTION AI (PLAN A UPDATED)")
    print(" Khoi dong thanh cong!")
    print(f" - CNN Pretrained Model: {config.MODEL_PATH}")
    print(f" - LiteRT Threads: {config.INFERENCE_THREADS} | Max CNN FPS: {config.CNN_MAX_FPS}")
    print(" - Nhan [R]: Reset bao dong khi benh nhan da an toan.")
    print(" - Nhan [Q] hoac [ESC]: Thoat chuong trinh.")
    print("=" * 70)

    window_name = "VinMec Smart Ward - Fall Detection AI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[THONG BAO] Nguon video da ket thuc hoac ngat ket noi.")
                break

        current_time = time.time()
        frame_count += 1
        dt_fps = current_time - prev_time
        if dt_fps >= 0.5:
            fps_display = frame_count / dt_fps
            frame_count = 0
            prev_time = current_time

        # Xử lý frame qua Fall Detector (kèm overlay y tế)
        frame, tele = detector.process(frame, current_time=current_time, fps_display=fps_display)

        cv2.imshow(window_name, frame)

        # Bắt phím bấm
        key = cv2.waitKey(playback_delay) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("[THONG BAO] Nguoi dung yeu cau dung chuong trinh.")
            break
        elif key in (ord("r"), ord("R")):
            detector.reset_alarm()
            print("[ACTION] Da Reset trang thai bao dong ve BINH THUONG (NORMAL).")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("[KET THUC] Chuong trinh giam sat da dong an toan.")


if __name__ == "__main__":
    video_source = 0
    if len(sys.argv) > 1:
        arg_val = " ".join(sys.argv[1:]).strip("\"'")
        for prefix in ["path\\to\\", "path/to/", "path\\to/", "path/to\\"]:
            if arg_val.lower().startswith(prefix):
                arg_val = arg_val[len(prefix):].strip("\"'")

        if arg_val.isdigit():
            video_source = int(arg_val)
        else:
            video_source = arg_val
        print(f"[KHOI DONG] Su dung nguon video tu: {video_source}")
    else:
        print("[KHOI DONG] Su dung Webcam mac dinh (index 0).")

    run_vinmec_fall_detection(source=video_source)
