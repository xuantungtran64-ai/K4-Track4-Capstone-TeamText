"""
Chốt 1: Bộ phân loại ảnh Fall / No Fall sử dụng mô hình CNN pretrained (TFLite / LiteRT).
Chạy trên CPU với tần suất tối ưu (CNN_MAX_FPS), ánh xạ nhãn chuẩn hóa và hỗ trợ crop người.
"""

import os
import time
from typing import Optional, Tuple
import cv2
import numpy as np

# Thử import runtime LiteRT ưu tiên, fallback sang TFLite hoặc TF
Interpreter = None
try:
    import ai_edge_litert.interpreter as litert
    Interpreter = litert.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            Interpreter = None


class FallImageClassifier:
    """
    Wrapper phân loại Fall / No Fall từ frame ảnh RGB.
    Hợp đồng: predict_fall_probability(frame, bbox) -> float trong [0.0, 1.0] hoặc None.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_threads: int = 2,
        max_fps: float = 8.0,
        input_mode: str = "landmark_crop",
        crop_margin: float = 0.15,
    ):
        self.model_path = model_path
        self.num_threads = num_threads
        self.max_fps = max_fps
        self.input_mode = input_mode
        self.crop_margin = crop_margin

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_loaded = False

        # Cache kết quả suy luận để không gọi quá giới hạn max_fps
        self.last_infer_time = 0.0
        self.last_fall_prob: Optional[float] = None
        self.last_raw_prediction: Optional[float] = None
        self.last_latency_ms: float = 0.0

        self._initialize_model()

    def _initialize_model(self):
        """Khởi tạo interpreter một lần duy nhất lúc startup."""
        if self.model_path and os.path.exists(self.model_path) and Interpreter is not None:
            try:
                self.interpreter = Interpreter(
                    model_path=self.model_path,
                    num_threads=self.num_threads,
                )
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.is_loaded = True
                print(f"[CNN] Khoi tao thanh cong mo hinh TFLite tai: {self.model_path} ({self.num_threads} threads)")
            except Exception as e:
                print(f"[CNN Canh bao] Khong the nap mo hinh TFLite: {e}. He thong se chay che do Pose Baseline.")
                self.is_loaded = False
        else:
            if not self.model_path or not os.path.exists(self.model_path):
                print(f"[CNN Thong bao] Mo hinh chua ton tai tai: {self.model_path}. Chay che do Pose Baseline.")
            elif Interpreter is None:
                print("[CNN Thong bao] Chua cai dat LiteRT/TFLite runtime. Chay che do Pose Baseline.")
            self.is_loaded = False

    def predict_fall_probability(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        current_time: Optional[float] = None,
    ) -> Optional[float]:
        """
        Dự đoán xác suất ngã P(fall) trong khoảng [0.0, 1.0].
        :param frame: Frame ảnh BGR gốc
        :param bbox: (xmin, ymin, xmax, ymax) chuẩn hóa [0, 1] nếu có
        :param current_time: Timestamp hiện tại
        :return: fall_probability (0.0 -> 1.0) hoặc None nếu không khả dụng
        """
        if not self.is_loaded or self.interpreter is None:
            return None

        if current_time is None:
            current_time = time.time()

        # Kiểm soát tần suất gọi CNN theo CNN_MAX_FPS để tiết kiệm tài nguyên CPU
        min_interval = 1.0 / max(1.0, self.max_fps)
        if (current_time - self.last_infer_time) < min_interval and self.last_fall_prob is not None:
            return self.last_fall_prob

        h, w = frame.shape[:2]

        # Chuẩn bị ảnh đầu vào: Landmark Crop có lề hoặc Full Frame
        target_img = frame
        if self.input_mode == "landmark_crop" and bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            pad_x = (xmax - xmin) * self.crop_margin
            pad_y = (ymax - ymin) * self.crop_margin
            x1 = max(0, int((xmin - pad_x) * w))
            y1 = max(0, int((ymin - pad_y) * h))
            x2 = min(w, int((xmax + pad_x) * w))
            y2 = min(h, int((ymax + pad_y) * h))

            if (x2 - x1) >= 24 and (y2 - y1) >= 24:
                target_img = frame[y1:y2, x1:x2]

        # Tiền xử lý theo kiểm tra audit của mô hình EfficientNet:
        # Input shape: [1, 224, 224, 3], RGB, float32 trong khoảng [0, 255]
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        input_tensor = np.expand_dims(resized.astype(np.float32), axis=0)

        input_idx = self.input_details[0]["index"]
        output_idx = self.output_details[0]["index"]

        self.interpreter.set_tensor(input_idx, input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(output_idx)

        t1 = time.perf_counter()
        self.last_latency_ms = (t1 - t0) * 1000.0

        raw_pred = float(output[0][0])
        # Ánh xạ nhãn: Mô hình gốc quy định output < 0.5 là Fall.
        # Do đó: P(fall) = 1.0 - raw_prediction
        fall_prob = float(np.clip(1.0 - raw_pred, 0.0, 1.0))

        self.last_infer_time = current_time
        self.last_raw_prediction = raw_pred
        self.last_fall_prob = fall_prob

        return fall_prob
