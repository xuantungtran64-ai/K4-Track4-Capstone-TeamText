# Kế hoạch triển khai Fall Detection real-time trên CPU

**Loại tài liệu:** Implementation plan / coding-agent specification  
**Mục tiêu:** Demo phát hiện ngã qua webcam/video, ưu tiên mô hình pretrained, chạy CPU, không huấn luyện model mới trong phạm vi hiện tại.  
**Nguyên tắc:** Giữ thay đổi tối thiểu với code sẵn có; đo đạc thay vì giả định hiệu năng; không coi một dự đoán đơn lẻ là sự kiện ngã đã được xác nhận.

---

## 1. Mục tiêu và phạm vi

Xây dựng một pipeline phát hiện ngã gồm hai chốt:

1. **Chốt 1 - phân loại ảnh:** dùng CNN pretrained đã được fine-tune cho Fall / No Fall để tìm frame có dấu hiệu đáng ngờ.
2. **Chốt 2 - xác minh thời gian:** sử dụng 10 frame gần nhất cùng landmarks MediaPipe để kiểm tra chuyển động đi xuống, thay đổi tư thế sang nằm ngang, và duy trì tư thế sau chuyển động.

MediaPipe Pose cũng là mô hình pretrained. Không tự xây dựng hay huấn luyện backbone, không huấn luyện một temporal neural network mới trong phiên bản demo này.

### Nằm trong phạm vi

- Đọc webcam real-time hoặc file video.
- MediaPipe Pose để trích xuất landmarks và tạo bounding box xấp xỉ.
- CNN Fall / No Fall pretrained, ưu tiên bản TensorFlow Lite / LiteRT.
- Bộ đệm vòng lưu ít nhất 10 frame đã xử lý và timestamp tương ứng.
- Bộ xác minh rule-based với ngưỡng cấu hình được.
- Overlay trạng thái và một số chỉ số debug.
- Kiểm thử logic, smoke test model, đo FPS/latency và ghi rõ các giới hạn.

### Không nằm trong phạm vi

- Huấn luyện hoặc fine-tune lại model.
- Multi-person tracking hoàn chỉnh.
- Cam kết dùng trong môi trường y tế hoặc hệ thống báo động an toàn thực tế.
- Thay đổi UI/framework dự án hiện có nếu không cần cho pipeline.
- Thêm detector nặng như YOLO chỉ để tạo bounding box, trừ khi code hiện tại chứng minh MediaPipe không đáp ứng.

---

## 2. Quyết định kiến trúc

```text
Webcam / video
      |
      v
Frame capture + timestamp
      |
      +------------------------------+
      |                              |
      v                              v
MediaPipe Pose (pretrained)     Frame RGB cho CNN
      |                              |
      v                              v
Landmarks + visibility          CNN Fall / No Fall (pretrained)
      |                              |
      +-------------+----------------+
                    |
                    v
         Ring buffer 10+ frame
                    |
                    v
      Fast candidate trigger
   (CNN đáng ngờ OR motion đáng ngờ)
                    |
                    v
        Chốt 2: Temporal verifier
  - chuyển động hông đi xuống nhanh
  - tư thế ngang / góc thân người
  - chất lượng landmarks đủ tốt
  - tư thế nằm duy trì sau chuyển động
                    |
                    v
       MONITORING / CANDIDATE /
      PENDING / FALL_CONFIRMED
```

**Quy tắc thiết kế quan trọng:** CNN là nguồn bằng chứng, không phải cổng chặn tuyệt đối. Nếu CNN không đánh dấu Fall nhưng chuyển động hình học rất đáng ngờ, pipeline vẫn phải được phép kiểm tra chốt 2. Cách này tránh việc một false negative ở CNN khiến sự kiện không bao giờ được xác minh.

Chốt 2 trong bản demo này là logic xác minh, không phải một model cần pretrained. Nếu sau này cần temporal neural model pretrained, có thể thử riêng mô hình phân loại chuỗi landmarks; không thay vào bản chính trước khi kiểm tra kích thước cửa sổ, giấy phép, preprocessing và chất lượng.

---

## 3. Chiến lược model pretrained

### 3.1 MediaPipe Pose

- Dùng API MediaPipe hiện có trong repository nếu đang chạy ổn; không thay toàn bộ API chỉ để làm mới code.
- Nếu cần triển khai mới, ưu tiên cấu hình model Lite / CPU; chỉ xử lý một người trong demo để kiểm soát chi phí tính toán.
- Nếu dùng legacy `mp.solutions.pose`, bắt đầu với `model_complexity=0`, bật smoothing khi phù hợp và giữ đúng luồng BGR -> RGB.
- Nếu dùng Pose Landmarker Tasks, chọn chế độ VIDEO hoặc LIVE_STREAM cho chuỗi video/camera và truyền timestamp tăng dần. Tài liệu chính thức lưu ý video/live mode có cơ chế tracking để giảm việc chạy model trên từng frame.
- Chỉ bật segmentation mask nếu thật sự cần, vì demo hiện tại chỉ cần landmarks.

Tài liệu: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python

### 3.2 CNN ảnh Fall / No Fall: model ứng viên ban đầu

**Model ứng viên:** `Siddhartha276/Fall_Detection`, file `fall_detection_model.tflite`.

- Model card mô tả một classifier nhị phân Fall / No Fall sử dụng EfficientNetB0, backbone khởi tạo từ ImageNet rồi fine-tune cho tác vụ ngã.
- File TFLite được liệt kê khoảng 4.55 MB; trang model công bố license MIT.
- Đây là model tác vụ cụ thể đã fine-tune, phù hợp hơn việc lấy một MobileNet/ImageNet classifier thuần rồi giả định đầu ra có nghĩa là Fall / No Fall.
- **Không được coi chất lượng của model là đã được xác minh:** model card không mô tả đầy đủ nguồn dữ liệu, độ đa dạng camera, hay kết quả benchmark độc lập. Phải kiểm tra model trên video demo của dự án.

Nguồn model: https://huggingface.co/Siddhartha276/Fall_Detection  
Nguồn code mô tả model/preprocessing: https://huggingface.co/Siddhartha276/Fall_Detection/blob/main/cnn.py

#### Việc agent bắt buộc phải làm trước khi tích hợp CNN

1. Tải model một lần vào thư mục `models/`; không tải lại từ mạng khi mỗi lần chạy ứng dụng.
2. Đọc metadata tensor: input shape, dtype, quantization, output shape, số output và interpreter version.
3. Đọc lại preprocessing của model và kiểm tra RGB/BGR, resize, scale/normalization. Không tự ý áp dụng normalization hai lần.
4. Xác minh nhãn output bằng ví dụ Fall và No Fall. Có một mâu thuẫn cần xử lý: model card minh họa `prediction < 0.15` là Fall, trong khi `cnn.py` công bố dùng `prediction < 0.5` là Fall. Không được chọn một ngưỡng chỉ dựa vào hai ví dụ này. Đọc đúng file `.tflite`, kiểm tra input/output tensor, preprocessing, ý nghĩa score và đối chiếu bằng các frame Fall/No Fall; sau đó mới chuẩn hóa kết quả thành `fall_probability` trong `[0, 1]`.
5. Tạo wrapper có hợp đồng rõ ràng: `predict_fall_probability(frame) -> float` trong khoảng `[0, 1]`.
6. Nếu không thể xác minh preprocessing/nhãn hoặc model cho dự đoán bất hợp lý trên ví dụ đơn giản, không tự huấn luyện model thay thế. Ghi lại blocker và chạy baseline MediaPipe + rule-based để kiểm tra phần còn lại.

#### Runtime trên CPU

- Ưu tiên `ai-edge-litert` nếu tương thích môi trường Python hiện tại; API TFLite trong TensorFlow là phương án tương thích khác nếu đã cài sẵn TensorFlow.
- Không cài đồng thời nhiều runtime nặng nếu một runtime đã hoạt động.
- Đặt số thread inference thành cấu hình, bắt đầu với 2 thread rồi benchmark.
- Khởi tạo interpreter/model đúng một lần lúc startup, không khởi tạo trong vòng lặp frame.

Tài liệu LiteRT Python / nền tảng: https://developers.google.com/edge/litert/overview

### 3.3 Những điều không được làm

- Không dùng MobileNetV3 pretrained ImageNet trực tiếp để kết luận Fall / No Fall. ImageNet weights không có nhãn nghiệp vụ này.
- Không tự thay CNN bằng YOLO hoặc một mạng nặng chỉ vì dễ tìm checkpoint.
- Không tự huấn luyện hoặc fine-tune model nếu ứng viên pretrained không đạt yêu cầu; báo kết quả và xin quyết định tiếp theo.
- Không hardcode ngưỡng `0.15` hay `0.5`: model card và `cnn.py` hiện không nhất quán về ngưỡng Fall. Ngưỡng nội bộ chỉ là giá trị khởi đầu; phải hiệu chỉnh trên clip validation của dự án sau khi xác minh hướng score và mapping nhãn.

---

## 4. Pipeline runtime chi tiết

### Bước 1 - Capture

- Đọc frame từ webcam hoặc video thông qua OpenCV.
- Lưu `timestamp_sec` tăng đơn điệu cho từng frame được xử lý. Với video có FPS hợp lệ, có thể khởi tạo timestamp từ frame index/FPS; khi chạy webcam, dùng đồng hồ monotonic.
- Resize frame inference nếu camera quá lớn; khởi điểm demo 640x480 là đủ để thử, nhưng giữ kích thước gốc cho hiển thị nếu cần.
- Không để tốc độ hiển thị bắt buộc bằng tốc độ inference; nếu CPU không theo kịp thì bỏ qua frame mới có kiểm soát, không tích lũy backlog vô hạn.

### Bước 2 - MediaPipe Pose

Trích xuất ít nhất các landmarks sau:

- Vai trái/phải: IDs 11, 12.
- Hông trái/phải: IDs 23, 24.
- Các keypoints còn lại chỉ cần để tạo bbox xấp xỉ nếu hữu ích.

Tính trung điểm:

`shoulder_x = (x_11 + x_12) / 2`  
`shoulder_y = (y_11 + y_12) / 2`  
`hip_x = (x_23 + x_24) / 2`  
`hip_y = (y_23 + y_24) / 2`

Dùng `visibility` để đánh giá chất lượng. Nếu vai/hông không đủ visibility, đánh dấu frame không hợp lệ cho phép tính đó; không coi tọa độ thiếu là số 0.

Tạo bbox từ các landmarks đủ tin cậy nếu dự án chưa có detector người. Vì đây là bbox xấp xỉ từ pose, không giả định nó chính xác như person detector. Nếu thiếu quá nhiều điểm, đặt `bbox_valid=False`.

### Bước 3 - CNN inference

- Chạy classifier trên frame RGB ở tần suất giới hạn, bắt đầu `CNN_MAX_FPS = 8` để kiểm tra CPU; thay đổi theo benchmark.
- Mặc định giữ hai chế độ input có thể chọn qua config: `full_frame` và `landmark_crop`. Vì chưa biết dữ liệu huấn luyện gốc của model có dùng ảnh toàn cảnh hay crop người, không kết luận crop nào tốt hơn nếu chưa thử.
- Nếu dùng crop, tạo crop có margin từ bbox landmarks, clamp vào biên ảnh và bỏ crop quá nhỏ/không hợp lệ.
- Lưu `p_fall_raw`, `p_fall`, `cnn_timestamp` và thời gian inference. Dùng output đã được xác minh từ bước audit model.
- Có thể làm mượt score bằng EMA hoặc trung bình 3 lần dự đoán gần nhất. Không đưa score cũ vào buffer như thể đó là dự đoán của frame hiện tại; giữ timestamp rõ ràng.

### Bước 4 - Tạo temporal feature mỗi frame

Mỗi record của ring buffer cần có tối thiểu:

```text
FrameRecord:
  timestamp_sec
  frame_index
  pose_valid
  keypoints: x, y, visibility
  shoulder_midpoint: x, y
  hip_midpoint: x, y
  bbox: x_min, y_min, x_max, y_max (nếu hợp lệ)
  bbox_aspect_ratio (nếu hợp lệ)
  torso_angle_deg (nếu hợp lệ)
  fall_probability (score gần nhất + timestamp riêng)
```

Giữ `deque(maxlen=10)` hoặc cấu trúc tương đương. Khi bắt đầu, chưa đủ 10 record thì chưa tính chuyển động theo cửa sổ; chỉ hiện trạng thái `WARMING_UP`.

### Bước 5 - Tính feature chuyển động

#### A. Độ dịch chuyển xuống của hông

Do tọa độ `y` của ảnh tăng từ trên xuống dưới:

`delta_y_hip = hip_y[t] - hip_y[t-9]`  
`delta_t = timestamp[t] - timestamp[t-9]`  
`vertical_speed = delta_y_hip / delta_t`

- `delta_y > 0` nghĩa là hông dịch xuống trong cửa sổ.
- Dùng timestamp thực để tính vận tốc; không giả định 10 frame luôn tương đương 0.3 giây.
- Tọa độ landmarks MediaPipe chuẩn hóa theo kích thước ảnh, nhưng vẫn chịu ảnh hưởng khoảng cách người-camera. Vì vậy nên tính thêm `hip_drop_body_ratio`, tức dịch chuyển hông chia cho chiều cao bbox ở giai đoạn đứng trước đó (nếu đã quan sát được giai đoạn đứng). Không dùng chiều cao bbox hiện tại đang co lại khi ngã làm mẫu số duy nhất.
- Nếu chưa có baseline đứng hợp lệ, dùng ngưỡng ảnh tương đối và đánh dấu chất lượng thấp hơn; không tự nâng confidence.

#### B. Bounding Box Aspect Ratio

`R = bbox_width / bbox_height = (x_max - x_min) / (y_max - y_min)`

`R > 1.2` là ngưỡng khởi đầu để coi bbox có xu hướng nằm ngang, không phải ngưỡng tối ưu đã được chứng minh. Bbox tạo từ landmarks dễ biến động theo che khuất và tay/chân; không dùng riêng feature này để xác nhận ngã.

#### C. Góc thân người so với phương thẳng đứng

Tính góc dựa trên trung điểm vai và hông:

`theta_deg = degrees(atan2(abs(shoulder_x - hip_x), abs(shoulder_y - hip_y)))`

Với định nghĩa này, thân gần thẳng đứng có góc gần 0 độ, thân gần nằm ngang có góc tiến về 90 độ. Dùng feature khi vai/hông đủ visibility.

#### D. Tư thế nằm duy trì

Sau khi có candidate, theo dõi thêm khoảng 0.3-0.5 giây để xác minh người vẫn có tư thế nằm/ngang. Phần này làm tăng độ trễ nhưng giúp giảm cảnh báo sai khi người cúi người hoặc ngồi xuống.

### Bước 6 - Logic candidate và xác minh

Không yêu cầu CNN phải dự đoán Fall mới được kích hoạt temporal verifier.

**Candidate trigger** khi thỏa ít nhất một trong các điều kiện:

- `fall_probability >= CNN_THRESHOLD`, hoặc
- chuyển động hông xuống bất thường theo cửa sổ 10 frame, hoặc
- tư thế chuyển nhanh từ gần thẳng đứng sang gần nằm ngang.

**Điều kiện xác minh ban đầu:**

1. Có bằng chứng chuyển động xuống nhanh hoặc chuyển tư thế đột ngột trong 10 frame trước.
2. Tư thế hiện tại có bằng chứng nằm ngang: `R > R_MIN` **hoặc** `theta > THETA_MIN`, với landmarks/bbox đủ tin cậy.
3. Tư thế nằm duy trì trong khoảng xác nhận sau candidate.
4. Chất lượng dữ liệu pose không thấp đến mức các phép tính không đáng tin.

Nếu chỉ có `fall_probability` cao nhưng không có chuyển động trong lịch sử, không xác nhận Fall ngay. Xem đó là `CANDIDATE` hoặc `UNCERTAIN` để tiếp tục theo dõi.

Nếu thiếu landmarks trong nhiều frame liên tiếp, không phát ra `FALL_CONFIRMED` chỉ từ tọa độ mặc định; trả về `POSE_UNAVAILABLE/UNCERTAIN` và tiếp tục theo dõi.

### Bước 7 - State machine và khử cảnh báo lặp

Các trạng thái:

- `WARMING_UP`: buffer chưa đủ 10 frame.
- `MONITORING`: không có dấu hiệu đáng ngờ.
- `CANDIDATE`: CNN hoặc chuyển động đề xuất kiểm tra sâu hơn.
- `PENDING_CONFIRMATION`: đang kiểm tra tư thế nằm duy trì.
- `FALL_CONFIRMED`: đã vượt qua điều kiện xác minh; phát cảnh báo một lần.
- `COOLDOWN`: chặn nhiều cảnh báo cho cùng một sự kiện trong thời gian ngắn.
- `UNCERTAIN`: pose/bbox không đủ tin cậy hoặc model không hoạt động đúng.

Dùng `COOLDOWN_SEC` để tránh tạo một sự kiện cho mỗi frame trong lúc người vẫn nằm. Chỉ trở về `MONITORING` khi thời gian cooldown kết thúc và trạng thái pose đã thay đổi/ổn định theo logic.

---

## 5. Config khởi đầu

Các giá trị dưới đây chỉ để bắt đầu chạy demo; agent phải tách chúng ra khỏi code và cho phép thay đổi trong `config.py` hoặc YAML. Không khẳng định đây là bộ ngưỡng tối ưu.

| Tham số | Giá trị khởi đầu | Ghi chú |
|---|---:|---|
| `BUFFER_SIZE` | 10 | Theo thiết kế ban đầu |
| `CNN_MAX_FPS` | 8 | Điều chỉnh theo tốc độ CPU |
| `CNN_THRESHOLD` | 0.65 | Chỉ dùng sau khi output model được ánh xạ thành `p_fall` đúng |
| `BBOX_R_MIN` | 1.2 | Bằng chứng phụ, cần tinh chỉnh |
| `TORSO_ANGLE_MIN_DEG` | 55 | Góc đo từ phương thẳng đứng; cần tinh chỉnh |
| `PERSISTENCE_SEC` | 0.4 | Thời gian xác minh sau candidate |
| `COOLDOWN_SEC` | 5.0 | Tránh cảnh báo lặp |
| `MIN_VISIBILITY` | 0.5 | Ngưỡng khởi đầu; cần kiểm tra trên camera |
| `CAMERA_SIZE` | 640 / 480 | Chỉ là cấu hình thử, không ép độ phân giải hiển thị |
| `INFERENCE_THREADS` | 2 | Benchmark thêm 1/2/4 nếu runtime hỗ trợ |

Bổ sung ngưỡng chuyển động (dịch chuyển hông, vận tốc và/hoặc tỷ lệ theo chiều cao cơ thể) vào config. Không gán ngưỡng cuối cùng khi chưa xem dữ liệu thực tế. Tất cả feature và ngưỡng dùng trong quyết định cần có chế độ debug để xem khi tune.

---

## 6. Gợi ý cấu trúc thư mục

Nếu repository đã có cấu trúc tương đương, hãy thích nghi với cấu trúc hiện tại thay vì tạo lại từ đầu.

```text
project/
  main.py
  config.py
  requirements.txt
  README.md
  models/
    fall_detection_model.tflite
  src/
    video_capture.py
    pose_estimator.py
    fall_image_classifier.py
    pose_features.py
    temporal_verifier.py
    fall_detection_pipeline.py
    overlay.py
    timing.py
  scripts/
    download_models.py
    inspect_fall_model.py
    benchmark_cpu.py
  tests/
    test_pose_features.py
    test_temporal_verifier.py
    test_state_machine.py
    test_model_smoke.py
    assets/
      README.md
  outputs/
    .gitkeep
```

Không bắt buộc phải chia nhỏ thành nhiều file nếu dự án hiện tại là demo nhỏ. Tuy nhiên phải tách ít nhất classifier, feature extraction, temporal verifier và config để logic có thể kiểm thử độc lập.

### Interface/hợp đồng cần có

- `PoseEstimator.process(frame, timestamp_sec) -> PoseResult`
- `FallImageClassifier.predict_fall_probability(frame) -> float | None`
- `extract_pose_features(pose_result, timestamp_sec) -> FrameRecord`
- `TemporalFallVerifier.update(record) -> DetectionState`
- `FallDetectionPipeline.process(frame, timestamp_sec) -> DetectionResult`
- `DetectionResult` chứa `state`, `fall_probability`, `reason_codes`, `features_debug`, `latency_ms`.

Các interface có thể đổi tên theo codebase, nhưng cần giữ trách nhiệm tương đương.

---

## 7. Kế hoạch thực thi theo từng phase

### Phase 0 - Audit repository và môi trường

1. Đọc README, requirements và entrypoint hiện có.
2. Xác định Python version, OS, MediaPipe API đang dùng và code nào đang chạy.
3. Kiểm tra webcam/video capture trước khi tích hợp model.
4. Không ghi đè chức năng hiện có; lập danh sách file dự kiến thay đổi.

**Output:** ghi chú ngắn về cấu trúc hiện tại, lệnh chạy demo và các phụ thuộc xung đột nếu có.

### Phase 1 - Audit/tải model pretrained

1. Tải `fall_detection_model.tflite` từ model nguồn được nêu ở mục 3.2.
2. Ghi URL nguồn, license, ngày tải và hash file nếu thuận tiện.
3. Đọc tensor metadata và kiểm tra model load/invoke thành công trên CPU.
4. Kiểm tra preprocessing, output shape, ý nghĩa output và ánh xạ nhãn bằng các ví dụ Fall/No Fall.
5. Đo latency inference trung bình và p95 trên máy demo.
6. Nếu model không qua smoke test, không được âm thầm huấn luyện hoặc đổi qua một checkpoint không rõ nguồn; tiếp tục xây dựng rule baseline và ghi rõ blocker.

**Output:** `inspect_fall_model.py`, log metadata, ghi chú preprocessing/label mapping, kết quả latency.

### Phase 2 - MediaPipe + webcam baseline

1. Dùng MediaPipe pretrained hiện có để lấy landmarks.
2. Vẽ skeleton/landmarks và điểm hông/vai; tạo bbox xấp xỉ nếu đủ landmarks.
3. Hiển thị FPS, pose validity và trạng thái `MONITORING`.
4. Kiểm tra camera bị mất, người ra khỏi frame và landmark không hợp lệ.

**Output:** một demo camera có overlay, chưa cần cảnh báo Fall chính xác.

### Phase 3 - Chốt 1: CNN classifier

1. Thêm class wrapper cho TFLite/LiteRT.
2. Dùng preprocessing đã xác minh; lưu `fall_probability` theo nghĩa nhất quán.
3. Giới hạn `CNN_MAX_FPS`, không gọi model mỗi frame nếu CPU quá tải.
4. Hiển thị xác suất và thời điểm dự đoán gần nhất để debug.
5. So sánh `full_frame` với `landmark_crop` trên các clip mẫu mà dự án có; chọn cấu hình có kết quả tốt hơn, không chọn dựa trên trực giác.

**Output:** CNN hoạt động độc lập và có thể bật/tắt bằng config.

### Phase 4 - Ring buffer và chốt 2

1. Lưu tối thiểu 10 `FrameRecord` gần nhất.
2. Viết phép tính chuyển động hông, aspect ratio, torso angle và posture persistence thành hàm độc lập.
3. Viết state machine gồm candidate, pending, confirmed, uncertain và cooldown.
4. Đảm bảo CNN false negative không chặn trigger do chuyển động.
5. Ghi `reason_codes`, ví dụ `FAST_DOWNWARD_MOTION`, `HORIZONTAL_POSTURE`, `POSTURE_PERSISTED`, để giải thích vì sao có/không có cảnh báo.

**Output:** Temporal verifier có unit tests, không phụ thuộc webcam.

### Phase 5 - Tích hợp real-time và tối ưu CPU

1. Nối capture -> Pose -> CNN -> buffer -> verifier -> overlay.
2. Khởi tạo tất cả model một lần.
3. Tách inference/display nếu cần; không xếp backlog frame vô hạn.
4. Giảm kích thước frame inference hoặc giảm tần suất CNN trước khi cân nhắc đổi model.
5. Đo FPS thực tế, thời gian MediaPipe, thời gian CNN và tổng latency.

**Output:** demo webcam/video chạy end-to-end với thông số benchmark.

### Phase 6 - Kiểm thử và bàn giao

1. Chạy unit tests.
2. Chạy smoke test model.
3. Thử video có ngã thật, nằm xuống có chủ đích, ngồi, cúi nhặt đồ, quỳ/ngồi xổm, che khuất một phần và người đi khỏi frame.
4. Ghi lại false positives, false negatives và latency theo từng sự kiện.
5. Cập nhật README: cài đặt, tải model, chạy webcam, chạy video, đổi threshold, xử lý lỗi camera/model và các giới hạn.

**Output:** mã chạy được, test report và hướng dẫn chạy độc lập.

---

## 8. Kiểm thử bắt buộc

### Unit tests cho chốt 2

Tạo chuỗi landmarks giả hoặc fixture tối thiểu cho các trường hợp:

1. Đứng yên 10 frame -> không có candidate Fall.
2. Ngồi xuống chậm -> không xác nhận Fall chỉ vì hông hạ thấp.
3. Cúi nhặt đồ -> không xác nhận nếu không chuyển sang tư thế nằm duy trì.
4. Ngã mô phỏng: hông đi xuống nhanh, thân chuyển ngang, giữ tư thế -> đi đến `FALL_CONFIRMED`.
5. Nằm sẵn từ đầu video -> CNN score cao nhưng thiếu chuyển động trước đó, không kết luận ngay rằng vừa xảy ra một cú ngã.
6. Landmarks mất/visibility thấp -> `UNCERTAIN`, không dùng tọa độ số 0 để tạo fall giả.
7. Timestamp không đều -> vận tốc tính bằng `delta_t` thực, không bằng giả định FPS cố định.
8. Một sự kiện kéo dài nhiều frame -> chỉ phát một cảnh báo trong cooldown.

### Video acceptance checklist

- [ ] Video có cú ngã: thời điểm candidate và confirm có thể quan sát được.
- [ ] Người tự nằm xuống: kiểm tra liệu hệ thống nhầm là ngã không.
- [ ] Người đứng lên/ngồi xuống/cúi người: đo false alarm.
- [ ] Che khuất hoặc camera nhìn nghiêng: kiểm tra trạng thái `UNCERTAIN`.
- [ ] Không tìm thấy pose: ứng dụng tiếp tục chạy, không crash.
- [ ] Thoát webcam bằng phím được quy định: giải phóng camera và model đúng cách.

Không chỉ báo accuracy theo frame. Với fall detection, báo cáo tối thiểu:

- Event recall (tỷ lệ sự kiện ngã được phát hiện).
- Event precision (tỷ lệ cảnh báo đúng).
- False alarms per hour nếu có đủ thời lượng video âm tính.
- Detection latency từ lúc bắt đầu cú ngã tới lúc xác nhận.
- FPS trung bình và p95 latency của MediaPipe, CNN và toàn pipeline.

Với demo nội bộ, chưa cần đặt target độ chính xác phi thực tế. Điều bắt buộc là agent báo số đo thật, mô tả clip đã thử và không tuyên bố “real-time” chỉ dựa vào việc camera mở được.

---

## 9. Hướng dẫn CPU/performance

- Chỉ xử lý một người bằng MediaPipe trong bản đầu.
- Giữ CNN ở 224x224 nếu đó là input đã xác minh của checkpoint.
- Dùng LiteRT/TFLite runtime có sẵn; không dựng một PyTorch runtime lớn chỉ để chạy model `.tflite`.
- Mặc định chạy Pose theo từng frame xử lý; cho phép bỏ frame khi xử lý chậm.
- Chạy CNN theo giới hạn FPS cấu hình; temporal feature rule-based được tính từ record đã có.
- Tắt segmentation và debug drawing nặng khi benchmark.
- Cache interpreter, tensor details và transform; không cấp phát object/model lớn không cần thiết trong mỗi frame.
- Đo hiệu năng trong chế độ release bình thường, không suy ra FPS từ số FPS báo bởi camera.

Nếu pipeline quá chậm theo thứ tự ưu tiên:

1. Giảm độ phân giải inference.
2. Giảm `CNN_MAX_FPS` nhưng giữ pose/time buffer ổn định.
3. Giảm số lần vẽ debug overlay.
4. Điều chỉnh số thread và đo lại.
5. Chỉ sau đó mới đề xuất đổi checkpoint; ghi rõ đánh đổi chất lượng/độ trễ trước khi đổi.

---

## 10. Tiêu chí hoàn thành (Definition of Done)

Agent chỉ báo hoàn tất khi tất cả mục áp dụng sau đã xong:

- [ ] Repository hiện có đã được audit; thay đổi không phá chức năng đang chạy.
- [ ] MediaPipe model chạy trên CPU và lấy được landmarks hợp lệ.
- [ ] CNN pretrained được tải một lần, có nguồn/license và metadata được ghi lại.
- [ ] Preprocessing, label mapping và đầu ra `fall_probability` được xác minh trên ví dụ.
- [ ] Buffer 10 frame dùng timestamp thực.
- [ ] Temporal verifier có điều kiện chuyển động, tư thế và persistence, có state machine.
- [ ] CNN false negative không khóa hoàn toàn bước xác minh.
- [ ] Unit tests và model smoke test chạy qua.
- [ ] Webcam/video demo chạy end-to-end, thoát ứng dụng an toàn.
- [ ] Có FPS, latency và kết quả test bằng clip thực, kèm false positives/false negatives đã thấy.
- [ ] README có lệnh setup/run và hướng dẫn sửa threshold.
- [ ] Không huấn luyện/fine-tune model mới, không tuyên bố model đã chính xác nếu chưa đo.

---

## 11. Prompt giao việc trực tiếp cho coding agent

> Bạn là coding agent thực thi dự án Fall Detection real-time chạy CPU. Hãy đọc repository, README và requirements trước khi sửa code. Giữ nguyên API/cấu trúc đang chạy nếu có thể; không rewrite toàn bộ dự án khi chưa cần.
>
> Mục tiêu là MediaPipe Pose pretrained + CNN ảnh Fall/No Fall pretrained + temporal verifier rule-based dùng 10 frame gần nhất. Dùng model ứng viên `https://huggingface.co/Siddhartha276/Fall_Detection`, file `fall_detection_model.tflite`, nhưng bắt buộc audit metadata, input preprocessing và output label mapping trước khi tin vào score. Lưu ý model card minh họa threshold `< 0.15` là Fall còn `cnn.py` dùng `< 0.5`; xác minh sự không nhất quán này trên đúng file `.tflite`, không sao chép ngưỡng mù quáng. Nếu model không qua smoke test, không tự huấn luyện và không âm thầm đổi model; ghi rõ lý do, vẫn hoàn thiện/test baseline MediaPipe + temporal rules.
>
> Thực hiện theo thứ tự: (1) audit repository/môi trường; (2) model smoke test; (3) MediaPipe webcam baseline; (4) CNN wrapper; (5) 10-frame buffer và temporal verifier; (6) tích hợp state machine/overlay; (7) unit tests, video test và CPU benchmark; (8) README.
>
> Quy tắc bắt buộc: mọi threshold nằm trong config; timestamp dùng để tính dt; landmarks thiếu phải được đánh dấu invalid thay vì dùng số 0; CNN không được là hard gate chặn motion trigger; một sự kiện không được tạo cảnh báo lặp mỗi frame; không download model khi ứng dụng đang chạy; không train/fine-tune bất kỳ model nào; không thêm dependency/model nặng nếu không cần.
>
> Sau khi triển khai, trả về báo cáo: file đã sửa, kiến trúc cuối, cách setup/run, kết quả tests, output mapping/preprocessing đã xác minh, FPS và latency đo được, các clip đã thử, false positives/false negatives quan sát được, blockers còn lại và lệnh tái hiện kết quả. Không bịa số đo hoặc claim chất lượng chưa kiểm tra.

---

## 12. Nguồn tham khảo model

1. [MediaPipe Pose Landmarker - tài liệu Python chính thức](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
2. [Fall image classifier ứng viên](https://huggingface.co/Siddhartha276/Fall_Detection) - EfficientNetB0 fine-tuned, bản TFLite, license MIT theo model card.
3. [Code của model ứng viên](https://huggingface.co/Siddhartha276/Fall_Detection/blob/main/cnn.py) - đối chiếu preprocessing và ánh xạ nhãn.
4. [LiteRT - runtime Python và nền tảng](https://developers.google.com/edge/litert/overview)

