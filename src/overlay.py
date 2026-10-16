"""
Giao diện Giám sát Bệnh viện Thông minh (Hospital CCTV UI Overlay).
Vẽ khung bao theo trạng thái, trục trọng tâm cơ thể, Telemetry Panel và cảnh báo SOS khẩn cấp.
"""

import time
from typing import Optional, Tuple
import cv2
import numpy as np

from src.temporal_verifier import DetectionResult
from src.pose_features import FrameRecord


def draw_text_with_bg(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.55,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
    padding: int = 4,
) -> int:
    """Vẽ chữ có nền màu tương phản sắc nét trên CCTV."""
    x, y = pos
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        img,
        (x - padding, y - h - padding),
        (x + w + padding, y + baseline + padding),
        bg_color,
        -1,
    )
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)
    return h + baseline + padding * 2


def render_hospital_overlay(
    frame: np.ndarray,
    result: DetectionResult,
    record: FrameRecord,
    fps: float = 0.0,
    cnn_latency_ms: float = 0.0,
) -> np.ndarray:
    """
    Vẽ toàn bộ overlay CCTV y tế lên frame ảnh.
    """
    h, w = frame.shape[:2]
    now = time.time()

    # Xác định màu sắc theo trạng thái FSM
    if result.state in ("MONITORING", "NORMAL"):
        theme_color = (0, 220, 0)      # Xanh lá
        status_text = "BINH THUONG (MONITORING)"
    elif result.state in ("CANDIDATE", "PENDING_CONFIRMATION", "SUSPECTED"):
        theme_color = (0, 165, 255)    # Vàng cam
        status_text = f"CANH BAO: MAT THANGBANG ({result.elapsed_fall_time:.1f}s)..."
    elif result.state == "COOLDOWN":
        theme_color = (255, 180, 0)    # Xanh dương / Vàng nhạt
        status_text = f"COOLDOWN DANG KICH HOAT ({result.elapsed_fall_time:.1f}s)"
    elif result.state in ("WARMING_UP", "UNCERTAIN"):
        theme_color = (180, 180, 180)  # Xám
        status_text = f"DANG QUET TIM BENH NHAN ({result.state})"
    else:
        theme_color = (0, 0, 255)      # Đỏ
        status_text = "BAO DONG: BENH NHAN NGA! (FALL CONFIRMED)"

    # 1. Vẽ Bounding Box quanh bệnh nhân
    if record.pose_valid and record.bbox is not None:
        xmin, ymin, xmax, ymax = record.bbox
        pad_x = int(record.box_w * w * 0.08)
        pad_y = int(record.box_h * h * 0.08)
        bx1 = max(0, int(xmin * w) - pad_x)
        by1 = max(0, int(ymin * h) - pad_y)
        bx2 = min(w, int(xmax * w) + pad_x)
        by2 = min(h, int(ymax * h) + pad_y)

        box_thick = 3 if result.state not in ("MONITORING", "NORMAL", "WARMING_UP") else 2
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), theme_color, box_thick)

        # Nhãn trên đầu
        label_y = max(24, by1 - 8)
        draw_text_with_bg(
            frame,
            f"BENH NHAN: {result.state}",
            (bx1, label_y),
            scale=0.52,
            text_color=(255, 255, 255),
            bg_color=theme_color,
            thickness=2,
        )

    # 2. Vẽ Trục Trọng Tâm Cơ Thể (Nối Hip -> Shoulder)
    if record.pose_valid and record.shoulder_midpoint and record.hip_midpoint:
        pt_sh = (int(record.shoulder_midpoint[0] * w), int(record.shoulder_midpoint[1] * h))
        pt_hip = (int(record.hip_midpoint[0] * w), int(record.hip_midpoint[1] * h))

        axis_color = (255, 255, 0) if result.state in ("MONITORING", "NORMAL") else (0, 0, 255)
        cv2.line(frame, pt_hip, pt_sh, axis_color, 3, cv2.LINE_AA)
        cv2.circle(frame, pt_sh, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pt_hip, 6, (255, 0, 255), -1, cv2.LINE_AA)

        # Góc nghiêng thân
        mid_x = (pt_sh[0] + pt_hip[0]) // 2 + 10
        mid_y = (pt_sh[1] + pt_hip[1]) // 2
        cv2.putText(
            frame,
            f"{int(record.torso_angle_deg)} deg",
            (mid_x, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 3. Telemetry Panel góc trái trên
    panel_w = 460
    panel_h = 195
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 80), 1)

    ty = 32
    # Tiêu đề hệ thống
    cv2.putText(
        frame,
        "VINMEC SMART WARD - 2-STAGE FALL AI",
        (20, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    ty += 26

    # Trạng thái
    cv2.putText(
        frame,
        f"Trang thai: {status_text}",
        (20, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        theme_color,
        2,
        cv2.LINE_AA,
    )
    ty += 24

    # Vận tốc rơi & Góc nghiêng
    cv2.putText(
        frame,
        f"Toc do roi (Vy): {result.vy:.2f} | Goc nghieng: {int(result.torso_angle)} deg",
        (20, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    ty += 24

    # Khung bao W/H & CNN P(fall)
    cnn_str = f"{result.fall_probability:.2f}" if result.fall_probability > 0 else "N/A"
    cv2.putText(
        frame,
        f"Ty le Khung bao (W/H): {result.aspect_ratio:.2f} | CNN P(fall): {cnn_str}",
        (20, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    ty += 24

    # FPS & Latency
    cv2.putText(
        frame,
        f"Hieu nang: {fps:.1f} FPS | CNN Latency: {cnn_latency_ms:.1f}ms",
        (20, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    ty += 24

    # Lý do / Reason Codes
    reasons_str = ", ".join(result.reason_codes[-2:]) if result.reason_codes else "NORMAL"
    cv2.putText(
        frame,
        f"Reason: {reasons_str}",
        (20, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )
    ty += 22

    # Phím bấm
    cv2.putText(
        frame,
        "Phim [R]: Reset Bao dong | Phim [Q]: Thoat",
        (20, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 255, 160),
        1,
        cv2.LINE_AA,
    )

    # 4. Hiển thị Timestamp CCTV góc phải
    time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame,
        f"CAM-WARD-01 | {time_str}",
        (w - 290, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # 5. Hiệu ứng Báo động khẩn cấp khi FALL_CONFIRMED
    if result.is_fall_confirmed:
        flash_on = (int(now * 4) % 2) == 0
        if flash_on:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 14)

        sos_w = int(w * 0.85)
        sos_h = 75
        sos_x1 = (w - sos_w) // 2
        sos_y1 = (h - sos_h) // 2
        sos_x2 = sos_x1 + sos_w
        sos_y2 = sos_y1 + sos_h

        cv2.rectangle(frame, (sos_x1, sos_y1), (sos_x2, sos_y2), (0, 0, 200), -1)
        cv2.rectangle(frame, (sos_x1, sos_y1), (sos_x2, sos_y2), (255, 255, 255), 3)

        msg_line1 = "SOS: PHAT HIEN BENH NHAN NGA!"
        msg_line2 = "DA BAO Y TA TRUC! Nhan [R] de Reset"

        (lw1, _), _ = cv2.getTextSize(msg_line1, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        (lw2, _), _ = cv2.getTextSize(msg_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)

        cv2.putText(
            frame,
            msg_line1,
            (sos_x1 + (sos_w - lw1) // 2, sos_y1 + 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            msg_line2,
            (sos_x1 + (sos_w - lw2) // 2, sos_y1 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (200, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame

