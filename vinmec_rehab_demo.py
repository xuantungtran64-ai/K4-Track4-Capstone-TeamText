"""
VINMEC TELE-REHABILITATION DEMO (OPTION 2A)
===========================================
Ứng dụng giám sát phục hồi chức năng khớp khuỷu tay (Elbow Flexion/Extension) từ xa.
Phát hiện sai tư thế bù trừ (shoulder compensation), đếm số lần tập đạt chuẩn và xuất log CSV.

Tối ưu phần cứng:
- CPU: Intel Core i5 (CPU-optimized, model_complexity=0 cho MediaPipe Pose).
- RAM: Quản lý bộ nhớ tối thiểu, phù hợp máy 8GB RAM.
- Mục tiêu: Đạt >= 25 FPS mượt mà trên CPU không cần GPU rời.
"""

import csv
import os
import sys
import time
import cv2
import numpy as np
import mediapipe as mp


def calculate_angle(a, b, c):
    """
    Tính góc 2D giữa 3 điểm a, b, c (với b là đỉnh góc).
    :param a: Tọa độ [x, y] điểm thứ nhất (ví dụ: Vai)
    :param b: Tọa độ [x, y] đỉnh góc (ví dụ: Khuỷu tay)
    :param c: Tọa độ [x, y] điểm thứ ba (ví dụ: Cổ tay)
    :return: Giá trị góc từ 0 đến 180 độ
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


class AngleSmoother:
    """
    Bộ lọc làm mượt góc sử dụng Exponential Moving Average (EMA).
    Chống hiện tượng rung giật số (jitter) khi camera bị nhiễu hoặc ánh sáng yếu.
    """

    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.smoothed_angle = None

    def update(self, current_angle):
        if self.smoothed_angle is None:
            self.smoothed_angle = current_angle
        else:
            self.smoothed_angle = (
                self.alpha * current_angle + (1.0 - self.alpha) * self.smoothed_angle
            )
        return self.smoothed_angle

    def reset(self):
        self.smoothed_angle = None


class VinmecRehabAnalyzer:
    """
    Bộ phân tích động học phục hồi chức năng khuỷu tay VinMec.
    Máy trạng thái hữu hạn (FSM) hỗ trợ bắt lỗi tư thế bù trừ và đo ROM.
    """

    def __init__(self, side="LEFT", model_complexity=0):
        self.side = side.upper()  # 'LEFT' hoặc 'RIGHT'
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=model_complexity,  # 0: Lite (CPU tối ưu), 1: Full
            smooth_landmarks=True,
        )

        # Bộ lọc làm mượt góc
        self.elbow_smoother = AngleSmoother(alpha=0.35)
        self.shoulder_smoother = AngleSmoother(alpha=0.35)

        # Biến trạng thái FSM tập luyện
        self.stage = "EXTENSION"  # 'EXTENSION' hoặc 'FLEXION'
        self.reps = 0
        self.best_rom = 180.0  # Biên độ gập tốt nhất (góc gập nhỏ nhất)
        self.current_rep_min_angle = 180.0
        self.is_bad_posture = False
        self.feedback = "Bat dau: Duoi thang tay de chuan bi"

        # Danh sách log buổi tập
        self.session_logs = []

    def toggle_side(self):
        """Chuyển đổi tay tập giữa bên TRÁI và bên PHẢI."""
        self.side = "RIGHT" if self.side == "LEFT" else "LEFT"
        self.elbow_smoother.reset()
        self.shoulder_smoother.reset()
        self.feedback = f"Da chuyen sang theo doi tay: {self.side}"
        return self.side

    def reset_counter(self):
        """Reset lại số lần tập và kỷ lục ROM."""
        self.reps = 0
        self.best_rom = 180.0
        self.current_rep_min_angle = 180.0
        self.stage = "EXTENSION"
        self.is_bad_posture = False
        self.feedback = "Da reset so lan tap. San sang bat dau!"
        self.elbow_smoother.reset()
        self.shoulder_smoother.reset()

    def process(self, frame, current_time=None):
        """
        Xử lý 1 frame hình ảnh và cập nhật trạng thái phân tích.
        :param frame: Frame hình ảnh BGR từ OpenCV
        :param current_time: Timestamp tùy chọn
        :return: (annotated_frame, telemetry_dict)
        """
        if current_time is None:
            current_time = time.time()

        h, w, _ = frame.shape

        # Tối ưu CPU Intel Core i5: Thu nhỏ nhẹ frame suy luận cho MediaPipe Pose
        if w > 320:
            scale_ratio = 320.0 / w
            infer_h = max(16, int(h * scale_ratio))
            infer_frame = cv2.resize(frame, (320, infer_h), interpolation=cv2.INTER_LINEAR)
        else:
            infer_frame = frame

        frame_rgb = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        has_detection = False
        visibility_ok = False
        raw_elbow_angle = 0.0
        smoothed_elbow_angle = 0.0
        smoothed_shoulder_drift = 0.0
        progress = 0.0

        if results.pose_landmarks:
            has_detection = True
            landmarks = results.pose_landmarks.landmark

            # Lấy mốc theo bên được chọn
            if self.side == "LEFT":
                lm_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                lm_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
                lm_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                lm_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            else:
                lm_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                lm_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                lm_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                lm_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

            # Bước 3.2: Visibility Guard - Kiểm tra độ tin cậy của 4 mốc
            visibilities = [
                lm_shoulder.visibility,
                lm_elbow.visibility,
                lm_wrist.visibility,
                lm_hip.visibility,
            ]
            min_vis = min(visibilities)

            if min_vis >= 0.5:
                visibility_ok = True

                pt_shoulder = [lm_shoulder.x, lm_shoulder.y]
                pt_elbow = [lm_elbow.x, lm_elbow.y]
                pt_wrist = [lm_wrist.x, lm_wrist.y]
                pt_hip = [lm_hip.x, lm_hip.y]

                # Bước 4: Tính toán góc khuỷu tay và góc vung bả vai
                raw_elbow_angle = calculate_angle(pt_shoulder, pt_elbow, pt_wrist)
                raw_shoulder_drift = calculate_angle(pt_hip, pt_shoulder, pt_elbow)

                # Làm mượt góc bằng EMA
                smoothed_elbow_angle = self.elbow_smoother.update(raw_elbow_angle)
                smoothed_shoulder_drift = self.shoulder_smoother.update(raw_shoulder_drift)

                # Quy đổi tiến trình gập tay ROM: 155 độ = 0%, 50 độ = 100%
                progress = float(np.interp(smoothed_elbow_angle, (50, 155), (100, 0)))
                progress = max(0.0, min(100.0, progress))

                # Logic 4.1: Bắt lỗi bù trừ vận động (shoulder drift > 35 độ)
                if smoothed_shoulder_drift > 35.0:
                    self.is_bad_posture = True
                    self.feedback = "SAI TU THE: Khep khuyu tay sat suon!"
                else:
                    self.is_bad_posture = False

                # Logic 4.2: Máy trạng thái đếm số lần lặp & ghi nhận ROM
                if not self.is_bad_posture:
                    # Ghi nhận góc gập nhỏ nhất trong lần tập hiện tại
                    if smoothed_elbow_angle > 20.0:
                        self.current_rep_min_angle = min(
                            self.current_rep_min_angle, smoothed_elbow_angle
                        )

                    # Trạng thái duỗi tay: Góc > 155 độ
                    if smoothed_elbow_angle > 155.0:
                        self.stage = "EXTENSION"
                        self.feedback = "Gap tay len tu tu (Muc tieu < 50 do)"
                        self.current_rep_min_angle = 180.0

                    # Trạng thái gập tay đạt chuẩn: Góc < 50 độ và chuyển từ EXTENSION
                    elif smoothed_elbow_angle < 50.0 and self.stage == "EXTENSION":
                        self.stage = "FLEXION"
                        self.reps += 1
                        self.best_rom = min(self.best_rom, smoothed_elbow_angle)
                        self.feedback = "Tot lam! Duoi tay xuong tu tu"

                        # Ghi log buổi tập vào bộ nhớ
                        log_time_str = time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(current_time)
                        )
                        log_entry = [
                            log_time_str,
                            self.reps,
                            round(smoothed_elbow_angle, 1),
                            round(self.best_rom, 1),
                            "VALID",
                        ]
                        self.session_logs.append(log_entry)
                        print(f"[REHAB] +1 Rep hop le! Tong: {self.reps} | Goc gap: {smoothed_elbow_angle:.1f} deg")

                # =========================================================
                # Bước 5.1: Vẽ trực quan hóa các khớp (Joint Visualizer)
                # =========================================================
                px_shoulder = (int(lm_shoulder.x * w), int(lm_shoulder.y * h))
                px_elbow = (int(lm_elbow.x * w), int(lm_elbow.y * h))
                px_wrist = (int(lm_wrist.x * w), int(lm_wrist.y * h))
                px_hip = (int(lm_hip.x * w), int(lm_hip.y * h))

                arm_color = (0, 100, 255) if self.is_bad_posture else (0, 255, 0)

                # Vẽ đường nối thân (Hip -> Shoulder) màu xám mờ
                cv2.line(frame, px_hip, px_shoulder, (120, 120, 120), 2, cv2.LINE_AA)
                # Vẽ đường cánh tay (Shoulder -> Elbow -> Wrist)
                cv2.line(frame, px_shoulder, px_elbow, arm_color, 4, cv2.LINE_AA)
                cv2.line(frame, px_elbow, px_wrist, arm_color, 4, cv2.LINE_AA)

                # Vẽ các khớp tròn bán kính 8px
                for pt in (px_shoulder, px_elbow, px_wrist):
                    cv2.circle(frame, pt, 8, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 6, arm_color, -1, cv2.LINE_AA)

                # Hiển thị số đo góc ngay cạnh khuỷu tay
                angle_text = f"{int(smoothed_elbow_angle)} deg"
                cv2.rectangle(
                    frame,
                    (px_elbow[0] + 10, px_elbow[1] - 18),
                    (px_elbow[0] + 100, px_elbow[1] + 6),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    angle_text,
                    (px_elbow[0] + 12, px_elbow[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Nếu sai tư thế, vẽ cảnh báo góc vung vai
                if self.is_bad_posture:
                    cv2.putText(
                        frame,
                        f"Lech vai: {int(smoothed_shoulder_drift)} deg (>35 deg)",
                        (px_shoulder[0] - 120, px_shoulder[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 50, 255),
                        2,
                        cv2.LINE_AA,
                    )

            else:
                # Visibility thấp (< 0.5)
                visibility_ok = False
                self.feedback = "CANH BAO: Vui long dungkinh/lui lai de camera thay ro canh tay!"
        else:
            self.feedback = "DANG QUET TIM BENH NHAN..."

        # =========================================================
        # Bước 5.2: Thanh tiến trình biên độ gập (ROM Progress Bar)
        # =========================================================
        bar_x = w - 65
        bar_y1 = 120
        bar_y2 = h - 80
        bar_h = bar_y2 - bar_y1

        # Nền thanh đo
        cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + 35, bar_y2), (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + 35, bar_y2), (200, 200, 200), 2)

        # Đổ đầy thanh đo theo progress
        fill_h = int((progress / 100.0) * bar_h)
        bar_color = (0, 100, 255) if self.is_bad_posture else (0, 220, 0)
        if fill_h > 0:
            cv2.rectangle(
                frame,
                (bar_x + 2, bar_y2 - fill_h),
                (bar_x + 33, bar_y2 - 2),
                bar_color,
                -1,
            )

        # Phần trăm ROM
        cv2.putText(
            frame,
            f"{int(progress)}%",
            (bar_x - 5, bar_y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "ROM",
            (bar_x, bar_y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # =========================================================
        # Bước 5.3: Bảng Dashboard Y tế VinMec (Top Left Overlay)
        # =========================================================
        dash_w = 460
        dash_h = 190
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (12 + dash_w, 12 + dash_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (12, 12), (12 + dash_w, 12 + dash_h), (80, 80, 80), 1)

        # Dòng 1: Tiêu đề hệ thống
        cv2.putText(
            frame,
            "VINMEC TELE-REHAB: PHUC HOI KHOP KHUYU TAY",
            (22, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 230, 255),
            2,
            cv2.LINE_AA,
        )

        # Dòng 2: Số lần đạt chuẩn (Valid Reps)
        cv2.putText(
            frame,
            f"So lan dat chuan (Valid Reps): {self.reps}",
            (22, 70),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Dòng 3: Biên độ gập tốt nhất (Best ROM) & Góc hiện tại
        rom_str = f"{int(self.best_rom)} deg" if self.best_rom < 179 else "--"
        cv2.putText(
            frame,
            f"Bien do tot nhat (Max ROM): {rom_str} | Goc hien tai: {int(smoothed_elbow_angle)} deg",
            (22, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Dòng 4: Hướng dẫn / Cảnh báo lỗi
        if self.is_bad_posture:
            fb_color = (0, 60, 255)  # Đỏ cam cảnh báo
        elif not visibility_ok and has_detection:
            fb_color = (0, 215, 255)  # Vàng
        elif self.stage == "FLEXION":
            fb_color = (0, 255, 120)  # Xanh ngọc
        else:
            fb_color = (0, 255, 0)  # Xanh lá

        cv2.putText(
            frame,
            f"Huong dan: {self.feedback}",
            (22, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            fb_color,
            2,
            cv2.LINE_AA,
        )

        # Dòng 5: Tay theo dõi và góc vung vai
        cv2.putText(
            frame,
            f"Tay theo doi: {self.side} | Lech vai: {int(smoothed_shoulder_drift)} deg (Max 35)",
            (22, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

        # Dòng 6: Hướng dẫn phím tắt
        cv2.putText(
            frame,
            "Phim [R]: Reset | Phim [S]: Doi tay Trai/Phai | Phim [Q]: Thoat & Luu CSV",
            (22, 182),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (150, 255, 150),
            1,
            cv2.LINE_AA,
        )

        telemetry = {
            "side": self.side,
            "reps": self.reps,
            "stage": self.stage,
            "elbow_angle": smoothed_elbow_angle,
            "shoulder_drift": smoothed_shoulder_drift,
            "best_rom": self.best_rom,
            "is_bad_posture": self.is_bad_posture,
            "visibility_ok": visibility_ok,
            "has_detection": has_detection,
            "feedback": self.feedback,
            "progress": progress,
        }

        return frame, telemetry

    def export_csv(self, filename="rehab_session_log.csv"):
        """
        Lưu danh sách lần tập đạt chuẩn ra file CSV.
        Định dạng: Timestamp,Rep_Number,Min_Angle_Achieved,Session_Best_ROM,Status
        """
        file_exists = os.path.exists(filename)
        with open(filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(filename) == 0:
                writer.writerow(
                    [
                        "Timestamp",
                        "Rep_Number",
                        "Min_Angle_Achieved",
                        "Session_Best_ROM",
                        "Status",
                    ]
                )
            for row in self.session_logs:
                writer.writerow(row)

        print(f"[XUAT BAO CAO] Da luu thanh cong {len(self.session_logs)} luot tap vao file: {filename}")

    def close(self):
        """Giải phóng tài nguyên Pose."""
        self.pose.close()


def run_vinmec_rehab_demo(source=0):
    """
    Hàm thực thi chính ứng dụng VinMec Tele-Rehabilitation Demo.
    :param source: 0 (Webcam mặc định) hoặc đường dẫn file video mẫu.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[LOI] Khong the mo nguon video: {source}")
        return

    is_webcam = isinstance(source, int)
    is_video_file = isinstance(source, str)

    # Đặt độ trễ playback delay chuẩn FPS nếu là file video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if is_video_file and video_fps and video_fps > 0:
        playback_delay = max(1, int(1000.0 / video_fps))
    else:
        playback_delay = 10

    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Khởi tạo analyzer tối ưu CPU Intel Core i5
    analyzer = VinmecRehabAnalyzer(side="LEFT", model_complexity=0)

    window_name = "VinMec Tele-Rehab - Elbow Flexion AI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    print("=" * 65)
    print(" VINMEC TELE-REHABILITATION DEMO (OPTION 2A)")
    print(" Khoi dong thanh cong!")
    print(" - Phim [R]: Reset so lan tap ve 0.")
    print(" - Phim [S]: Chuyen doi tay tap Trai/Phai (LEFT/RIGHT).")
    print(" - Phim [Q] hoac [ESC]: Thoat chuong trinh & Luu log CSV.")
    print("=" * 65)

    prev_time = time.time()
    fps_display = 0.0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[THONG BAO] Nguon video da ket thuc.")
                break

        current_time = time.time()
        frame_count += 1
        dt_fps = current_time - prev_time
        if dt_fps >= 0.5:
            fps_display = frame_count / dt_fps
            frame_count = 0
            prev_time = current_time

        # Đối với Webcam: Lật gương (cv2.flip 1) để bệnh nhân quan sát tự nhiên
        if is_webcam:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # Xử lý frame qua Rehab Analyzer
        frame, tele = analyzer.process(frame, current_time=current_time)

        # Hiển thị FPS góc phải trên
        cv2.putText(
            frame,
            f"FPS: {fps_display:.1f} | CPU-Optimized",
            (w - 230, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, frame)

        # Xử lý phím tắt
        key = cv2.waitKey(playback_delay) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("[THONG BAO] Nguoi dung yeu cau thoat chuong trinh.")
            break
        elif key in (ord("r"), ord("R")):
            analyzer.reset_counter()
            print("[ACTION] Da reset so lan tap.")
        elif key in (ord("s"), ord("S")):
            new_side = analyzer.toggle_side()
            print(f"[ACTION] Da doi sang theo doi tay: {new_side}")

    cap.release()
    analyzer.export_csv(filename="rehab_session_log.csv")
    analyzer.close()
    cv2.destroyAllWindows()
    print("[KET THUC] Chuong trinh Tele-Rehab da ket thuc thanh cong.")


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
        print(f"[KHOI DONG] Su dung nguon video: {video_source}")
    else:
        print("[KHOI DONG] Su dung Webcam mac dinh (index 0).")

    run_vinmec_rehab_demo(source=video_source)

