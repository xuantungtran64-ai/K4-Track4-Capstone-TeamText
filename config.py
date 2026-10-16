"""
Cấu hình hệ thống VinMec Fall Detection (Plan A Updated)
Tất cả các ngưỡng vật lý, tham số mô hình và cấu hình runtime đều được khai báo tại đây.
"""

import os

# Đường dẫn mô hình
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fall_detection_model.tflite")
MODEL_URL = "https://huggingface.co/Siddhartha276/Fall_Detection/resolve/main/fall_detection_model.tflite"

# Cấu hình Chốt 1: CNN Classifier (TFLite / LiteRT)
CNN_MAX_FPS = 8.0               # Giới hạn tần suất gọi CNN trên CPU (khởi điểm 8 FPS)
CNN_THRESHOLD = 0.65            # Ngưỡng xác suất P(fall) để kích hoạt candidate
CNN_INPUT_MODE = "landmark_crop" # 'landmark_crop' (crop người có lề) hoặc 'full_frame'
INFERENCE_THREADS = 2           # Số luồng CPU cho LiteRT (benchmark: 2 threads tối ưu nhất ~67ms)
CROP_MARGIN = 0.15              # Phần trăm mở rộng biên khi crop người

# Cấu hình Chốt 2: Temporal Verifier & Buffer
BUFFER_SIZE = 10                # Độ dài bộ đệm trượt (ít nhất 10 frame gần nhất)
BBOX_R_MIN = 1.2                # Tỷ lệ khung bao W/H tối thiểu để coi là nằm ngang
TORSO_ANGLE_MIN_DEG = 55.0      # Góc nghiêng thân người so với phương thẳng đứng (độ)
RAPID_DROP_VY = 0.45            # Vận tốc rơi dọc hông (chiều cao khung hình / giây)
PERSISTENCE_SEC = 0.4           # Thời gian xác minh tư thế nằm duy trì sau candidate (giây)
COOLDOWN_SEC = 5.0              # Thời gian chống lặp cảnh báo sau một sự kiện ngã (giây)
MIN_VISIBILITY = 0.5            # Độ tin cậy tối thiểu của MediaPipe Landmark

# Cấu hình MediaPipe Pose
MODEL_COMPLEXITY = 0            # 0: Lite (CPU tối ưu), 1: Full
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
SMOOTH_LANDMARKS = True
SCALE_INFER_WIDTH = 320         # Thu nhỏ frame suy luận Pose để đạt >= 35 FPS trên CPU

# Cấu hình Camera & Video
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

