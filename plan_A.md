# AGENT IMPLEMENTATION PLAN: VINMEC ELDERLY FALL DETECTION DEMO (OPTION 2B)

## 1. Mục tiêu & Ràng buộc Kỹ thuật (Constraints)
- **Mục tiêu:** Xây dựng hệ thống Camera AI giám sát an toàn cho bệnh nhân cao tuổi tại phòng bệnh/hành lang bệnh viện VinMec. Hệ thống phát hiện té ngã theo thời gian thực với bộ lọc chống báo giả (anti-false-positive khi bệnh nhân chỉ cúi nhặt đồ hoặc ngồi xuống ghế từ từ).
- **Phần cứng mục tiêu:** Chạy mượt (>= 25 FPS) trên CPU phổ thông (Intel Core i5, 8GB RAM).
- **Thư viện sử dụng:** `opencv-python`, `mediapipe`, `numpy`, `collections` (`deque`).
- **Kiến trúc file:** Tạo 1 file duy nhất `vinmec_fall_demo.py` có khả năng đọc từ Webcam (`0`) hoặc đọc từ file video mẫu nếu truyền đường dẫn qua tham số dòng lệnh (`python vinmec_fall_demo.py [video_path]`).

---

## 2. Quy trình Thực thi Từng bước (Step-by-Step Execution)

### Bước 1: Thiết lập Môi trường (Environment Setup)
1. Kiểm tra và cài đặt các gói:
   `pip install opencv-python mediapipe numpy`
2. Tạo file `vinmec_fall_demo.py`.

### Bước 2: Thiết kế Cấu trúc Bộ đệm Thời gian (Temporal Sliding Window)
Để phân biệt "ngã đột ngột" với "từ từ ngồi/nằm xuống", cần lưu lịch sử chuyển động trong `N = 10` frames gần nhất (~0.33 giây ở 30 FPS):
- Sử dụng `collections.deque(maxlen=10)` để lưu trữ:
  - `hip_y_history`: Lịch sử tọa độ trục dọc $Y$ của tâm hông.
  - `time_history`: Lịch sử thời gian thực `time.time()` của từng frame.

### Bước 3: Trích xuất Đặc trưng Hình học Cơ thể (Biomechanical Feature Extraction)
Khởi tạo `mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)`.
Trong mỗi frame, nếu phát hiện được `pose_landmarks`:
1. **Tâm Vai (`shoulder_center`):** Trung bình cộng tọa độ `(x, y)` của `LEFT_SHOULDER` (11) và `RIGHT_SHOULDER` (12).
2. **Tâm Hông (`hip_center`):** Trung bình cộng tọa độ `(x, y)` của `LEFT_HIP` (23) và `RIGHT_HIP` (24).
3. **Khung bao cơ thể (Dynamic Bounding Box):**
   - Lấy danh sách tất cả các điểm mốc có `visibility > 0.5`.
   - Tính `xmin, xmax = min(x_list), max(x_list)` và `ymin, ymax = min(y_list), max(y_list)`.
   - Chiều rộng khung bao: `box_w = xmax - xmin`.
   - Chiều cao khung bao: `box_h = ymax - ymin`.

### Bước 4: Tính toán 3 Chỉ số Vật lý Phát hiện Ngã (Core Fall Metrics)
Cài đặt chính xác 3 công thức sau trong mỗi frame:

1. **Chỉ số 1 - Vận tốc rơi dọc của trọng tâm (`vertical_velocity` - $v_y$):**
   - Đưa `hip_center[1]` vào `hip_y_history` và `time.time()` vào `time_history`.
   - Nếu `len(hip_y_history) >= 5`:
     - `dt = time_history[-1] - time_history[0]`
     - `dy = hip_y_history[-1] - hip_y_history[0]` (Lưu ý: trục Y của ảnh hướng xuống dưới, nên rơi xuống nghĩa là `dy > 0`).
     - `vy = dy / (dt + 1e-5)` (Đơn vị: tỷ lệ màn hình / giây).
   - Ngưỡng kích hoạt tốc độ rơi cao: `vy > 0.45` (Rơi gần nửa chiều cao khung hình trong 1 giây).

2. **Chỉ số 2 - Góc nghiêng trục thân người so với phương thẳng đứng (`torso_angle` - $\theta_{\text{torso}}$):**
   - Vector thân người từ Hông lên Vai: `dx = abs(shoulder_center[0] - hip_center[0])`, `dy = abs(hip_center[1] - shoulder_center[1])`.
   - Công thức góc nghiêng so với trục dọc:
     `torso_angle = np.degrees(np.arctan2(dx, dy + 1e-5))`
   - Ý nghĩa: Khi đứng thẳng `torso_angle ≈ 0° - 15°`. Khi ngã nằm ngang `torso_angle > 55°`.

3. **Chỉ số 3 - Tỷ lệ khung bao (`aspect_ratio` - $R$):**
   - Công thức: `aspect_ratio = box_w / (box_h + 1e-5)`.
   - Ý nghĩa: Khi đứng `aspect_ratio < 0.6`. Khi nằm gục hoặc ngã ngang sàn `aspect_ratio > 1.05`.

### Bước 5: Máy Trạng Thái Đa Tầng Chống Báo Giả (Multi-Stage Anti-False-Positive FSM)
Hệ thống hoạt động qua 3 trạng thái rõ ràng (`status`):

1. **Trạng thái 1: `BINH THUONG (NORMAL)` (Màu Xanh Lá)**
   - Trạng thái mặc định khi `torso_angle < 45` và `aspect_ratio < 0.9`.
   - Reset bộ đếm thời gian nằm sàn: `fall_start_time = None`.

2. **Trạng thái 2: `NGHI NGO NGA / MAT THANGBANG (SUSPECTED)` (Màu Vàng Cam)**
   - **Điều kiện kích hoạt:** Xảy ra 1 trong 2 trường hợp:
     - *Trường hợp A (Rơi nhanh):* `vy > 0.45` (Trọng tâm tụt đột ngột).
     - *Trường hợp B (Tư thế nằm):* (`torso_angle > 55` HOẶC `aspect_ratio > 1.05`) HOẶC (`shoulder_center[1] > 0.80` - đầu/vai tụt sát đáy khung hình).
   - **Hành động:**
     - Nếu `fall_start_time is None`: Gán `fall_start_time = time.time()`.
     - Tính thời gian đã duy trì tư thế ngã: `elapsed_fall_time = time.time() - fall_start_time`.
     - Hiển thị cảnh báo vàng: `"CANH BAO: Phat hien mat thang bang ({elapsed_fall_time:.1f}s)..."`.

3. **Trạng thái 3: `BAO DONG: BENH NHAN NGA! (FALL CONFIRMED)` (Màu Đỏ Chớp Nháy)**
   - **Điều kiện xác nhận:** Đang ở trạng thái `SUSPECTED` (thỏa mãn tư thế nằm: `torso_angle > 50` hoặc `aspect_ratio > 1.0` hoặc `shoulder_center[1] > 0.80`) liên tục trong **`elapsed_fall_time >= 1.2` giây** (Giúp loại bỏ hoàn toàn trường hợp bệnh nhân chỉ cúi gập người nhặt đồ rồi đứng dậy ngay).
   - **Hành động:**
     - Bật cờ `is_fall_confirmed = True`.
     - Vẽ khung viền đỏ dày `12px` nhấp nháy quanh toàn bộ màn hình.
     - Vẽ hộp thông báo khẩn cấp giữa màn hình: `"SOS: PHAT HIEN BENH NHAN NGA! DA BAO Y TA TRUC!"`.

### Bước 6: Thiết kế Giao diện Giám sát Bệnh viện (Hospital CCTV UI Overlay)
1. **Vẽ Bounding Box thông minh quanh người:**
   - Chuyển `(xmin, ymin, xmax, ymax)` sang tọa độ pixel.
   - Vẽ khung chữ nhật bao quanh bệnh nhân với màu theo trạng thái (Xanh lá / Vàng / Đỏ) kèm nhãn trạng thái phía trên đầu.
2. **Vẽ Trục Trọng Tâm (Torso Axis Line):**
   - Kẻ 1 đường thẳng nổi bật nối từ `hip_center` tới `shoulder_center` để giám khảo thấy rõ góc nghiêng cơ thể thay đổi trực quan khi ngã.
3. **Bảng Thông số Kỹ thuật Góc Trái (Telemetry Panel):**
   - Dòng 1: `VINMEC SMART WARD - FALL DETECTION AI`
   - Dòng 2: `Trang thai: {status}`
   - Dòng 3: `Toc do roi (Vy): {vy:.2f} | Goc nghieng: {int(torso_angle)} deg`
   - Dòng 4: `Ty le Khung bao (W/H): {aspect_ratio:.2f}`
   - Dòng 5: `Phim [R]: Reset Bao dong | Phim [Q]: Thoat`

---

## 3. Tiêu chí Nghiệm thu (Definition of Done)
- [ ] Chạy `python vinmec_fall_demo.py` mở được webcam và vẽ đúng Bounding Box + trục nối Hông-Vai.
- [ ] Khi người dùng chỉ cúi nhẹ hoặc nghiêng người rồi đứng thẳng dậy ngay (< 1.2 giây), hệ thống chỉ hiện màu Vàng (`SUSPECTED`) rồi tự về Xanh lá (`NORMAL`), không báo động giả.
- [ ] Khi người dùng ngã nghiêng/nằm xuống hoặc tụt khuất xuống dưới khung hình quá 1.2 giây, màn hình lập tức chớp đỏ `FALL CONFIRMED`.
- [ ] Nhấn phím `r` có thể xóa trạng thái báo động khẩn cấp để demo lại nhiều lần.