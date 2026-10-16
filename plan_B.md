# AGENT IMPLEMENTATION PLAN: VINMEC TELE-REHABILITATION DEMO (OPTION 2A)

## 1. Mục tiêu & Ràng buộc Kỹ thuật (Constraints)
- **Mục tiêu:** Xây dựng ứng dụng Desktop real-time qua Webcam đo góc khớp khuỷu tay (Elbow Flexion/Extension) cho bệnh nhân phục hồi chức năng sau mổ của VinMec, đếm số lần tập đúng chuẩn, phát hiện sai tư thế bù trừ (compensation) và xuất log CSV.
- **Phần cứng mục tiêu:** Chạy mượt (>= 25 FPS) trên CPU phổ thông (Intel Core i5, tích hợp GPU, 8GB RAM), KHÔNG sử dụng mô hình nặng cần CUDA.
- **Thư viện sử dụng:** `opencv-python`, `mediapipe`, `numpy`. (Tất cả xử lý bằng tiếng Việt không dấu trên UI OpenCV để tránh lỗi font).
- **Kiến trúc file:** Tạo 1 file chạy chính duy nhất `vinmec_rehab_demo.py` (Single-file architecture để dễ debug và chạy ngay) kèm 1 file xuất dữ liệu tự động `rehab_session_log.csv`.

---

## 2. Quy trình Thực thi Từng bước (Step-by-Step Execution)

### Bước 1: Thiết lập Môi trường (Environment Setup)
1. Kiểm tra và cài đặt các gói cần thiết:
   `pip install opencv-python mediapipe numpy`
2. Tạo file `vinmec_rehab_demo.py`.

### Bước 2: Xây dựng các Hàm Toán học & Xử lý Tín hiệu (Math & Signal Utils)
Trong `vinmec_rehab_demo.py`, cài đặt 2 công cụ toán học sau:

1. **Hàm tính góc 2D giữa 3 điểm `calculate_angle(a, b, c)`:**
   - Đầu vào: `a`, `b`, `c` là tọa độ `[x, y]` của 3 khớp (với `b` là đỉnh góc).
   - Công thức:
     ```python
     a, b, c = np.array(a), np.array(b), np.array(c)
     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
     angle = np.abs(radians * 180.0 / np.pi)
     if angle > 180.0:
         angle = 360.0 - angle
     return angle
     ```
2. **Bộ lọc làm mượt góc `AngleSmoother` (Exponential Moving Average - EMA):**
   - Mục đích: Chống hiện tượng giật số (jitter) của MediaPipe khi ánh sáng yếu.
   - Công thức: `smoothed_angle = alpha * current_angle + (1 - alpha) * prev_smoothed_angle` với `alpha = 0.35`.

### Bước 3: Trích xuất Điểm mốc & Kiểm tra Độ tin cậy (Landmark Extraction & Visibility Guard)
Khởi tạo `mp.solutions.pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1)`.
Mỗi frame (sau khi `cv2.flip(frame, 1)`):
1. Lấy tọa độ chuẩn hóa `[x, y]` và `visibility` của 4 khớp bên TRÁI (hoặc PHẢI tùy chọn bằng phím bấm, mặc định là TRÁI):
   - `LEFT_SHOULDER` (ID: 11)
   - `LEFT_ELBOW` (ID: 13)
   - `LEFT_WRIST` (ID: 15)
   - `LEFT_HIP` (ID: 23)
2. **Logic Kiểm tra điều kiện nhìn thấy (Visibility Guard):**
   - Nếu bất kỳ khớp nào trong 4 khớp trên có `visibility < 0.5`:
     - Không cập nhật máy trạng thái đếm.
     - Hiển thị thông báo lên UI: `"CANH BAO: Vui long dungkinh/lui lai de camera thay ro canh tay!"` (Màu vàng).

### Bước 4: Thiết kế Máy Trạng Thái (Finite State Machine - FSM) & Bắt lỗi Tư thế
Nếu `visibility >= 0.5`, tính 2 góc đã làm mượt bằng EMA:
- `elbow_angle` = `calculate_angle(SHOULDER, ELBOW, WRIST)` (Góc gập khuỷu tay).
- `shoulder_drift` = `calculate_angle(HIP, SHOULDER, ELBOW)` (Góc vung bả vai).

**Logic 4.1 - Bắt lỗi bù trừ vận động (Compensatory Movement Detection):**
- Điều kiện lỗi: `shoulder_drift > 35.0` độ (bệnh nhân vung cùi chỏ ra xa sườn để ăn gian lực).
- Hành động khi lỗi:
  - Gán `is_bad_posture = True`.
  - Gán `feedback = "SAI TU THE: Khep khuyu tay sat suon!"`.
  - Đổi màu thanh tiến trình sang Đỏ/Cam `(0, 100, 255)`.
  - **KHÔNG** tăng biến đếm `reps` nếu đang trong trạng thái lỗi.

**Logic 4.2 - Máy trạng thái Đếm số lần lặp (Repetition Counter) & Đo ROM:**
- Chỉ chạy khi `is_bad_posture == False`:
  - Khởi tạo ban đầu: `stage = "EXTENSION"`, `reps = 0`, `best_rom = 180.0` (góc gập nhỏ nhất đạt được).
  - Cập nhật `best_rom = min(best_rom, elbow_angle)` khi `elbow_angle > 20` (loại bỏ nhiễu).
  - **Chuyển sang trạng thái Duỗi (`EXTENSION`):**
    - Khi `elbow_angle > 155.0`:
      - `stage = "EXTENSION"`
      - `feedback = "Gap tay len tu tu (Muc tieu < 50 do)"`
  - **Chuyển sang trạng thái Gập đạt chuẩn (`FLEXION`):**
    - Khi `elbow_angle < 50.0` VÀ `stage == "EXTENSION"`:
      - `stage = "FLEXION"`
      - `reps += 1`
      - `feedback = "Tot lam! Duoi tay xuong tu tu"`
      - Ghi 1 dòng vào bộ nhớ log: `[timestamp, reps, round(elbow_angle, 1), round(best_rom, 1), "VALID"]`.

### Bước 5: Xây dựng Giao diện Hiển thị (OpenCV UI Overlay)
Vẽ trực tiếp lên `frame` (kích thước `640x480` hoặc `1280x720`):
1. **Vẽ trực quan hóa tại khớp (Joint Visualizer):**
   - Nối đoạn thẳng đậm màu xanh lá `(0, 255, 0)` (hoặc đỏ nếu sai tư thế) giữa `SHOULDER -> ELBOW -> WRIST`.
   - Vẽ vòng tròn bán kính `8px` tại 3 khớp.
   - Hiển thị số đo góc `f"{int(elbow_angle)} deg"` ngay cạnh tọa độ pixel của `ELBOW`.
2. **Thanh tiến trình biên độ (Dynamic ROM Progress Bar):**
   - Quy đổi: `progress = np.interp(elbow_angle, (50, 155), (100, 0))` (Giới hạn trong khoảng 0 - 100%).
   - Vẽ hình chữ nhật rỗng bên phải màn hình và đổ đầy theo `progress`.
3. **Bảng Dashboard Y tế VinMec (Góc trên bên trái):**
   - Vẽ nền đen bán trong suốt (`cv2.addWeighted`).
   - Dòng 1: `VINMEC TELE-REHAB: PHUC HOI KHOP KHUYU TAY` (Màu vàng Cyan).
   - Dòng 2: `So lan dat chuan (Valid Reps): {reps}` (Chữ to, màu trắng).
   - Dòng 3: `Bien do gap tot nhat (Max Flexion ROM): {int(best_rom)} deg`.
   - Dòng 4: `Huong dan: {feedback}` (Màu xanh lá nếu đúng, Đỏ/Cam nếu sai tư thế).
   - Dòng 5 (Góc dưới): `Phim [R]: Reset | Phim [S]: Doi tay Trai/Phai | Phim [Q]: Thoat & Luu CSV`.

### Bước 6: Phím tắt Tương tác & Xuất Báo cáo (Keybindings & CSV Export)
Trong vòng lặp `cv2.waitKey(10)`:
- Nhấn `'r'`: Reset `reps = 0`, `best_rom = 180.0`.
- Nhấn `'s'`: Đổi biến `side = "RIGHT"` hoặc `"LEFT"` (chuyển đổi giữa bộ landmark `LEFT_*` và `RIGHT_*`).
- Nhấn `'q'`: Thoát vòng lặp, tự động lưu danh sách log ra file `rehab_session_log.csv` gồm các cột `Timestamp,Rep_Number,Min_Angle_Achieved,Session_Best_ROM,Status` và in thông báo hoàn tất ra terminal.

---

## 3. Tiêu chí Nghiệm thu (Definition of Done)
- [ ] Chạy lệnh `python vinmec_rehab_demo.py` lên cửa sổ webcam ngay lập tức không báo lỗi.
- [ ] Gập tay chuẩn (khép nách) từ `>155 độ` xuống `<50 độ` tăng đúng 1 Rep.
- [ ] Gập tay nhưng dang khuỷu tay rộng ra ngoài (`>35 độ`) hiện cảnh báo đỏ và không tăng Rep.
- [ ] Nhấn `q` sinh ra file `rehab_session_log.csv` hợp lệ.