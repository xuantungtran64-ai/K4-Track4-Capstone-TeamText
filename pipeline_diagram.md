# VINMEC AI MEDICAL HUB — KIẾN TRÚC PIPELINE PHÁT HIỆN TÉ NGÃ (PLAN A UPDATED)

Tài liệu này mô tả chi tiết kiến trúc kỹ thuật của hệ thống **Camera AI giám sát an toàn và phát hiện té ngã bệnh nhân cao tuổi (Two-Stage Real-Time Fall Detection Pipeline)** thông qua các biểu đồ luồng dữ liệu (Flowchart), máy trạng thái (State Diagram) và biểu đồ tuần tự (Sequence Diagram).

---

## 1. Biểu đồ Luồng Xử lý Toàn diện (System Pipeline Architecture)

Biểu đồ dưới đây minh họa luồng đi của dữ liệu từ thiết bị thu nhận hình ảnh qua **Chốt 1 (CNN Classifier)** và **Chốt 2 (MediaPipe Pose + Temporal Verifier)** đến giao diện giám sát bệnh viện:

```mermaid
flowchart TD
    %% Nguồn dữ liệu đầu vào
    subgraph S1["1. THU NHẬN HÌNH ẢNH (INPUT STAGE)"]
        A["Nguồn Video / Webcam (OpenCV)<br/><i>Resolution: 640x480 | Timestamp: Monotonic</i>"]
    end

    %% Tách 2 nhánh xử lý
    A --> B["Downscale Inference (320px)<br/><i>Tối ưu CPU Intel Core i5</i>"]
    A --> C["Frame gốc BGR / RGB<br/><i>224x224 cho CNN</i>"]

    %% Nhánh MediaPipe Pose
    subgraph S2["2. TRÍCH XUẤT ĐẶC TRƯNG HÌNH HỌC (CHỐT 2 PREPARATION)"]
        B --> D["MediaPipe Pose Lite<br/><i>model_complexity = 0 (42.4 FPS)</i>"]
        D --> E["33 Keypoints + Visibility Check<br/><i>Lọc mốc Vai & Hông: vis >= 0.5</i>"]
        E --> F["Tính toán đặc trưng cơ thể:<br/>• Tâm Vai & Tâm Hông<br/>• Bounding Box & Aspect Ratio (R)<br/>• Góc nghiêng trục thân (Torso Angle)"]
    end

    %% Nhánh CNN Classifier
    subgraph S3["3. PHÂN LOẠI ẢNH BẰNG CNN (CHỐT 1: EVIDENCE TRIGGER)"]
        C --> G{"Throttling Check<br/><i>dt >= 1/CNN_MAX_FPS (8 FPS)?</i>"}
        G -- "Đủ thời gian" --> H["Crop người có lề (Landmark Crop)<br/><i>hoặc Full Frame</i>"]
        H --> I["EfficientNetB0 TFLite (CPU 2 Threads)<br/><i>Latency: ~72ms | P(fall) = 1.0 - raw_pred</i>"]
        G -- "Chưa đủ thời gian" --> J["Dùng lại Cache P(fall) gần nhất<br/><i>Không tiêu tốn tài nguyên CPU</i>"]
    end

    %% Tích hợp vào FrameRecord & Ring Buffer
    F --> K["Tạo FrameRecord<br/><i>(Timestamp, Landmarks, R, Torso Angle, P_fall)</i>"]
    I --> K
    J --> K

    subgraph S4["4. BỘ ĐỆM THỜI GIAN & BỘ XÁC MINH (CHỐT 2: TEMPORAL VERIFIER)"]
        K --> L["Sliding Ring Buffer<br/><i>collections.deque(maxlen = 10)</i>"]
        L --> M{"Fast Candidate Trigger?<br/>• Rơi nhanh (Vy > 0.45)<br/>• CNN nghi ngờ (P >= 0.65)<br/>• Mất thăng bằng (Angle > 55°)"}
        
        M -- "Không có dấu hiệu" --> N["Trạng thái: MONITORING (NORMAL)<br/><i>Reset bộ đếm candidate</i>"]
        M -- "Thỏa mãn ít nhất 1 dấu hiệu" --> O["Trạng thái: CANDIDATE / PENDING<br/><i>Bắt đầu đếm thời gian duy trì</i>"]
        
        O --> P{"Xác minh tư thế nằm sàn duy trì<br/>• elapsed_time >= 0.8s<br/>• Tư thế nằm thật sự: R > 1.05 hoặc Hông sát sàn"}
        P -- "Đứng thẳng dậy (< 0.8s)" --> Q["Phục hồi: MONITORING<br/><i>Triệt tiêu báo động giả khi cúi nhặt đồ</i>"]
        P -- "Duy trì nằm sàn (>= 0.8s)" --> R["XÁC NHẬN TÉ NGÃ: FALL_CONFIRMED<br/><i>Kích hoạt chuông & viền đỏ SOS</i>"]
    end

    subgraph S5["5. CHỐNG BÁO ĐỘNG LẶP & ĐIỀU KHIỂN (COOLDOWN & ACTIONS)"]
        R --> S["Kích hoạt Cooldown (5.0s)<br/><i>Chặn gửi cảnh báo lặp cho cùng 1 sự kiện</i>"]
        S --> T{"Bệnh nhân an toàn?"}
        T -- "Nhấn phím [R] Reset" --> N
        T -- "Tự đứng dậy sau cooldown" --> N
    end

    subgraph S6["6. GIAO DIỆN GIÁM SÁT Y TẾ (CCTV UI OVERLAY)"]
        N --> U["Render Overlay CCTV:<br/>• Bounding Box đổi màu theo trạng thái<br/>• Kẻ trục Hông-Vai & nhãn góc nghiêng<br/>• Bảng Telemetry Panel góc trái<br/>• Hộp cảnh báo khẩn cấp SOS giữa màn hình"]
        O --> U
        R --> U
        S --> U
    end

    %% Định dạng style màu sắc
    style S1 fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    style S2 fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style S3 fill:#0f172a,stroke:#f59e0b,stroke-width:2px,color:#f8fafc
    style S4 fill:#1e1e38,stroke:#8b5cf6,stroke-width:2px,color:#f8fafc
    style S5 fill:#2d1515,stroke:#ef4444,stroke-width:2px,color:#f8fafc
    style S6 fill:#111827,stroke:#06b6d4,stroke-width:2px,color:#f8fafc
```

---

## 2. Biểu đồ Máy Trạng Thái Hữu Hạn (FSM State Transition Diagram)

Vòng đời chuyển đổi trạng thái của bệnh nhân được quản lý chặt chẽ theo Finite State Machine:

```mermaid
stateDiagram-v2
    [*] --> WARMING_UP: Khởi động hệ thống

    WARMING_UP --> MONITORING: Đã thu thập đủ 10 frames hợp lệ
    WARMING_UP --> UNCERTAIN: Thiếu mốc nhận diện / visibility < 0.5

    UNCERTAIN --> MONITORING: Mốc cơ thể hiển thị rõ trở lại

    MONITORING --> CANDIDATE: Phát hiện rơi nhanh (Vy > 0.45) HOẶC CNN P(fall) >= 0.65 HOẶC Mất thăng bằng (Angle > 55°)
    MONITORING --> UNCERTAIN: Mất dấu bệnh nhân

    CANDIDATE --> PENDING_CONFIRMATION: Duy trì dấu hiệu nghi ngờ ở frame tiếp theo
    CANDIDATE --> MONITORING: Người đứng thẳng dậy bình thường ngay lập tức

    PENDING_CONFIRMATION --> FALL_CONFIRMED: Duy trì tư thế nằm sàn liên tục >= 0.8s
    PENDING_CONFIRMATION --> MONITORING: Cúi gập người rồi đứng thẳng dậy (< 0.8s) -> Anti-False-Positive

    FALL_CONFIRMED --> COOLDOWN: Đã kích hoạt báo động khẩn cấp SOS
    COOLDOWN --> COOLDOWN: Vẫn nằm sàn trong thời gian cooldown (< 5.0s) -> Chặn lặp cảnh báo
    COOLDOWN --> MONITORING: Y tá nhấn phím [R] Reset HOẶC Bệnh nhân tự đứng dậy an toàn
```

---

## 3. Biểu đồ Tuần Tự Xử lý 1 Frame (Frame Execution Sequence)

Quy trình đồng bộ và xử lý song song có điều tiết giữa các thành phần trong 1 chu kỳ frame:

```mermaid
sequenceDiagram
    autonumber
    actor Cam as Camera / Video
    participant Pipe as FallDetectionPipeline
    participant Pose as MediaPipe Pose Lite
    participant Feat as Pose Feature Extractor
    participant CNN as FallImageClassifier (TFLite)
    participant Verif as TemporalFallVerifier (Buffer & FSM)
    participant UI as Hospital CCTV Overlay

    Cam->>Pipe: Đọc frame ảnh BGR (Timestamp t)
    
    rect rgb(20, 30, 48)
        Note over Pipe,Pose: Bước 1: Trích xuất mốc cơ thể (Real-time CPU)
        Pipe->>Pipe: Resize nhẹ frame (320px)
        Pipe->>Pose: Gửi frame RGB 320px
        Pose-->>Pipe: 33 Landmarks + Visibility
        Pipe->>Feat: extract_pose_features(landmarks, t)
        Feat-->>Pipe: FrameRecord (Tâm vai, hông, Box W/H, Torso Angle)
    end

    rect rgb(35, 30, 20)
        Note over Pipe,CNN: Bước 2: Chốt 1 - Phân loại ảnh CNN (Throttled 8 FPS)
        alt Đã trôi qua >= 125ms kể từ lần gọi CNN trước
            Pipe->>CNN: predict_fall_probability(frame, bbox, t)
            CNN->>CNN: Crop người 224x224 & Inference 2 threads (~72ms)
            CNN-->>Pipe: Xác suất P(fall) mới
        else Chưa đủ thời gian throttle (< 125ms)
            Pipe->>CNN: Lấy cached P(fall)
            CNN-->>Pipe: P(fall) đã lưu trong cache
        end
    end

    rect rgb(30, 20, 40)
        Note over Pipe,Verif: Bước 3: Chốt 2 - Xác minh thời gian & FSM
        Pipe->>Verif: update(FrameRecord có kèm P_fall)
        Verif->>Verif: Cập nhật Ring Buffer (10 frames)
        Verif->>Verif: Tính Vy theo dt thực tế & Kiểm tra Persistence
        Verif-->>Pipe: DetectionResult (State, is_fall_confirmed, Reason Codes)
    end

    rect rgb(15, 30, 35)
        Note over Pipe,UI: Bước 4: Hiển thị giao diện CCTV Bệnh viện
        Pipe->>UI: render_hospital_overlay(frame, DetectionResult, FrameRecord)
        UI-->>Pipe: Frame đã vẽ Telemetry, Trục Hông-Vai & SOS
        Pipe-->>Cam: Hiển thị lên màn hình OpenCV (>= 40-48 FPS)
    end
```

---

## 4. Bảng Tra cứu Mã Quyết định (Decision Reason Codes)

Khi hệ thống đưa ra quyết định, trường `reason_codes` ghi nhận chuỗi bằng chứng kỹ thuật để phục vụ việc giải trình và gỡ lỗi (Explainable AI):

| Mã quyết định (`reason_code`) | Ý nghĩa kỹ thuật | Trạng thái kích hoạt |
|---|---|---|
| `BUFFER_WARMING_UP` | Chưa thu thập đủ 10 frames ban đầu trong bộ đệm | `WARMING_UP` |
| `FAST_DOWNWARD_MOTION` | Vận tốc rơi dọc hông $v_y > 0.45\text{ screen/s}$ | `CANDIDATE` |
| `HORIZONTAL_POSTURE` | Tỷ lệ khung bao $R > 1.05$ hoặc vai/hông sát sàn ($y > 0.80$) | `CANDIDATE / PENDING` |
| `CNN_SUSPECTED` | Mô hình CNN TFLite dự đoán xác suất ngã $P(\text{fall}) \ge 0.65$ | `CANDIDATE / PENDING` |
| `CHECKING_POSTURE_PERSISTENCE` | Đang theo dõi người bệnh có duy trì tư thế nằm sau candidate | `PENDING_CONFIRMATION` |
| `POSTURE_PERSISTED_FALL_CONFIRMED` | Tư thế nằm sàn duy trì liên tục $\ge 0.8\text{s}$ $\rightarrow$ Xác nhận ngã | `FALL_CONFIRMED` |
| `COOLDOWN_ACTIVE` | Đang trong thời gian 5.0s ngăn phát cảnh báo lặp cho cùng sự kiện | `COOLDOWN` |
| `RECOVERED_STANDING` | Người bệnh đã tự đứng thẳng dậy ($R < 0.9$ và $\theta < 45^\circ$) | `MONITORING` |
| `LOW_VISIBILITY_OR_NO_POSE` | Mốc vai/hông có `visibility < 0.5` hoặc người khuất khỏi camera | `UNCERTAIN` |

