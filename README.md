# VINMEC AI MEDICAL HUB — 2-STAGE FALL DETECTION (PLAN A UPDATED)

Hệ thống Camera AI giám sát an toàn và phát hiện té ngã bệnh nhân cao tuổi thời gian thực tại phòng bệnh/hành lang bệnh viện VinMec, tối ưu hóa chạy mượt mà trên CPU phổ thông (Intel Core i5, 8GB RAM) với tốc độ **>= 40-48 FPS**.

---

## 1. Kiến trúc Hệ thống 2 Chốt (Two-Stage Pipeline)

```text
Webcam / Video Input
       |
       v
Frame capture + timestamp (monotonic)
       |
       +-------------------------------+
       |                               |
       v                               v
MediaPipe Pose Lite (pretrained)  Frame RGB / Landmark Crop
       |                               |
       v                               v
Landmarks + visibility check      CNN Fall / No Fall (TFLite pretrained)
       |                               |
       +---------------+---------------+
                       |
                       v
            Ring Buffer (10+ frames)
                       |
                       v
             Fast Candidate Trigger
       (CNN đáng ngờ OR motion đáng ngờ)
                       |
                       v
         Chốt 2: Temporal Verifier
       - Vận tốc rơi dọc hông (vy, delta_t thực)
       - Tư thế nằm ngang (aspect ratio R, torso angle theta)
       - Độ tin cậy visibility (loại bỏ landmark giả)
       - Tư thế nằm duy trì (persistence >= 0.8s)
                       |
                       v
          Finite State Machine (FSM)
   (WARMING_UP / MONITORING / CANDIDATE /
    PENDING_CONFIRMATION / FALL_CONFIRMED / COOLDOWN)
                       |
                       v
           Hospital CCTV UI Overlay
```

### Nguyên tắc thiết kế:
1. **Chốt 1 - Phân loại ảnh (CNN Pretrained):** Dùng mô hình EfficientNetB0 fine-tuned cho tác vụ Fall Detection (`Siddhartha276/Fall_Detection`), định dạng TFLite (~4.55 MB). Chạy giới hạn `CNN_MAX_FPS = 8.0` trên 2 CPU threads để tiết kiệm tài nguyên.
2. **Chốt 2 - Xác minh thời gian (Temporal Verifier):** Dùng MediaPipe Pose Lite trích xuất 33 mốc cơ thể và lưu 10 frames gần nhất. CNN đóng vai trò là nguồn bằng chứng, không phải cổng chặn tuyệt đối: nếu CNN không báo Fall nhưng chuyển động hình học bất thường (rơi nhanh, hông tụt đột ngột), hệ thống vẫn kích hoạt xác minh.
3. **Chống báo động giả:** Cúi gập người nhặt đồ hoặc ngồi xuống ghế từ từ không bao giờ kích hoạt báo động khẩn cấp do kiểm tra tư thế nằm sàn duy trì (`persistence_sec`).

---

## 2. Cấu trúc Thư mục

```text
K4-Track4-Capstone-TeamText/
├── config.py                   # Tập trung tất cả tham số, ngưỡng vật lý và cấu hình runtime
├── vinmec_fall_demo.py         # Entrypoint chính chạy Demo Phát hiện Té ngã
├── vinmec_rehab_demo.py        # Entrypoint Luồng B (Phục hồi chức năng)
├── vinmec_demo_app.py          # Launcher GUI điều khiển trung tâm
├── test_fall_logic.py          # Test suite tổng hợp toàn diện
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── models/
│   └── fall_detection_model.tflite # Mô hình TFLite pretrained
├── src/
│   ├── pose_features.py        # Trích xuất đặc trưng hình học & FrameRecord
│   ├── fall_image_classifier.py# Wrapper suy luận TFLite / LiteRT
│   ├── temporal_verifier.py    # Bộ xác minh thời gian & FSM
│   ├── fall_detection_pipeline.py # Pipeline tích hợp 2 chốt
│   └── overlay.py              # Giao diện giám sát CCTV bệnh viện
├── scripts/
│   ├── download_models.py      # Tải mô hình từ Hugging Face
│   ├── inspect_fall_model.py   # Kiểm tra metadata, tensor và đo độ trễ
│   └── benchmark_cpu.py        # Đo FPS và latency chi tiết trên CPU
└── tests/
    ├── test_model_smoke.py     # Smoke test mô hình TFLite
    ├── test_pose_features.py   # Unit test trích xuất landmarks
    ├── test_temporal_verifier.py # 8 kịch bản bắt buộc theo Plan A Updated
    └── test_state_machine.py   # Kiểm tra vòng đời máy trạng thái FSM
```

---

## 3. Cài đặt Môi trường (Setup)

1. Kích hoạt môi trường Python (Python 3.10+):
```bash
pip install -r requirements.txt
```

2. Tải và kiểm tra mô hình pretrained TFLite:
```bash
python scripts/download_models.py
python scripts/inspect_fall_model.py
```

---

## 4. Hướng dẫn Chạy Demo

### Cách 1: Chạy trực tiếp qua Webcam (Mặc định)
```bash
python vinmec_fall_demo.py
# Hoặc chạy file batch:
run_demo.bat
```

### Cách 2: Chạy với Video mẫu
```bash
python vinmec_fall_demo.py 20240912_101331.mp4
# Hoặc chạy file batch:
run_demo.bat 20240912_101331.mp4
```

### Cách 3: Giao diện Web Streamlit Đẳng cấp & Chỉn chu (Khuyên dùng)
```bash
streamlit run app_streamlit.py
# Hoặc nhấp đúp file batch:
run_streamlit.bat
```
*(Giao diện Streamlit cung cấp đầy đủ Video trực tiếp, biểu đồ thời gian thực, bảng Telemetry, điều chỉnh tham số động và tài liệu kiến trúc y tế).*

### Cách 4: Giao diện đồ họa Desktop Launcher UI
```bash
python vinmec_demo_app.py
# Hoặc chạy file batch:
run_ui.bat
```

### Phím tắt điều khiển khi chạy:
- **`[R]`**: Khôi phục hệ thống về trạng thái bình thường (Reset cảnh báo sau khi bệnh nhân đã được y tá hỗ trợ an toàn).
- **`[Q]`** hoặc **`[ESC]`**: Thoát chương trình an toàn.

---

## 5. Cấu hình & Tinh chỉnh Ngưỡng (`config.py`)

Tất cả các ngưỡng đều được quản lý tại file `config.py`:

| Tham số | Giá trị mặc định | Giải thích |
|---|---|---|
| `BUFFER_SIZE` | `10` | Số lượng frame lưu trong bộ đệm trượt |
| `CNN_MAX_FPS` | `8.0` | Tần suất tối đa gọi CNN classifier trên CPU |
| `CNN_THRESHOLD` | `0.65` | Ngưỡng xác suất P(fall) kích hoạt candidate |
| `CNN_INPUT_MODE` | `'landmark_crop'` | Chế độ crop theo khung người hoặc full_frame |
| `BBOX_R_MIN` | `1.05` | Tỷ lệ khung bao W/H tối thiểu nhận diện nằm ngang |
| `TORSO_ANGLE_MIN_DEG` | `55.0` | Góc nghiêng trục thân so với phương thẳng đứng |
| `RAPID_DROP_VY` | `0.45` | Ngưỡng vận tốc rơi dọc của hông (khung hình / s) |
| `PERSISTENCE_SEC` | `0.8` | Thời gian duy trì tư thế nằm trước khi chớp đỏ cảnh báo |
| `COOLDOWN_SEC` | `5.0` | Thời gian ngăn cảnh báo lặp cho cùng một sự kiện |
| `INFERENCE_THREADS` | `2` | Số luồng CPU chạy LiteRT |

---

## 6. Kết quả Kiểm thử & Đo đạc Thực tế (Benchmark)

### Kết quả đo trên CPU Intel Core i5:
- **Tốc độ toàn bộ Pipeline tích hợp:** **44.5 - 48.7 FPS** (Mục tiêu >= 25 FPS -> Đạt xuất sắc).
- **Độ trễ trung bình mỗi frame:** **20.51 ms / frame**.
- **MediaPipe Pose Lite đơn lẻ:** **42.4 FPS** (~23.58 ms).
- **CNN TFLite Classifier (2 threads):** **72.38 ms** (được giới hạn tần suất 8 FPS).

### Kiểm thử Unit Tests tự động:
Chạy lệnh:
```bash
python test_fall_logic.py
```
**Kết quả: Đạt 100% tất cả các bài kiểm tra:**
1. [PASS] Đứng thẳng bình thường (`NORMAL` - Xanh lá).
2. [PASS] Cúi gập người nhặt đồ (< 0.8s) -> `SUSPECTED` (Vàng), **không báo động giả**.
3. [PASS] Đứng dậy sau khi cúi -> Tự động khôi phục về `NORMAL`.
4. [PASS] Ngã mô phỏng và duy trì tư thế nằm -> Kích hoạt `FALL_CONFIRMED` (Chớp Đỏ + SOS).
5. [PASS] Phím Reset `[R]` -> Xóa báo động và đưa hệ thống về `NORMAL`.
6. [PASS] Smoke test mô hình TFLite (Input: `[1, 224, 224, 3]`, Output: `[1, 1]`, mapped $P(\text{fall}) \in [0, 1]$).
7. [PASS] Đầy đủ 8 kịch bản bắt buộc của Chốt 2 (Ngồi chậm, thiếu landmarks, timestamp không đều, cooldown một lần cảnh báo...).
