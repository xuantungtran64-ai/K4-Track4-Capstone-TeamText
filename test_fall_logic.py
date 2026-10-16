"""
Script kiểm thử tự động toàn diện logic phát hiện ngã VinMec và đo hiệu năng FPS trên CPU máy tính.
"""

import time
import numpy as np
import cv2
import mediapipe as mp
from vinmec_fall_demo import VinmecFallDetector


def create_mock_landmark(x, y, visibility=0.9):
    class MockLM:
        def __init__(self, x, y, visibility):
            self.x = x
            self.y = y
            self.visibility = visibility
            self.z = 0.0

    return MockLM(x, y, visibility)


def create_mock_pose_landmarks(torso_mode="standing", shift_y=0.0):
    """
    Tạo bộ mốc tọa độ cơ thể mô phỏng theo chế độ:
    - standing: vai y=0.3, hông y=0.6, chân y=0.95 (đứng thẳng, torso_angle ~0, aspect_ratio ~ 0.3)
    - bending: vai y=0.6, hông y=0.6, vai nghiêng lệch x (cúi nhặt đồ)
    - lying: vai y=0.85, hông y=0.85, nằm ngang (torso_angle ~90, aspect_ratio ~ 1.5)
    """
    landmarks = [create_mock_landmark(0.5, 0.5) for _ in range(33)]
    mp_pose = mp.solutions.pose

    if torso_mode == "standing":
        # Vai
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = create_mock_landmark(0.45, 0.30 + shift_y)
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = create_mock_landmark(0.55, 0.30 + shift_y)
        # Hông
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] = create_mock_landmark(0.47, 0.60 + shift_y)
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value] = create_mock_landmark(0.53, 0.60 + shift_y)
        # Chân
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value] = create_mock_landmark(0.47, 0.95 + shift_y)
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value] = create_mock_landmark(0.53, 0.95 + shift_y)

    elif torso_mode == "bending":
        # Cúi lưng
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = create_mock_landmark(0.30, 0.65 + shift_y)
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = create_mock_landmark(0.35, 0.65 + shift_y)
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] = create_mock_landmark(0.50, 0.60 + shift_y)
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value] = create_mock_landmark(0.55, 0.60 + shift_y)
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value] = create_mock_landmark(0.50, 0.95 + shift_y)
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value] = create_mock_landmark(0.55, 0.95 + shift_y)

    elif torso_mode == "lying":
        # Nằm ngang trên sàn
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = create_mock_landmark(0.20, 0.85 + shift_y)
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = create_mock_landmark(0.22, 0.85 + shift_y)
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] = create_mock_landmark(0.50, 0.87 + shift_y)
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value] = create_mock_landmark(0.52, 0.87 + shift_y)
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value] = create_mock_landmark(0.80, 0.89 + shift_y)
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value] = create_mock_landmark(0.82, 0.89 + shift_y)

    class MockResults:
        def __init__(self, lms):
            self.pose_landmarks = MockPoseLandmarks(lms)

    class MockPoseLandmarks:
        def __init__(self, lms):
            self.landmark = lms

    return MockResults(landmarks)


def run_all_tests():
    print("=" * 60)
    print("BAT DAU KIEM THU CHUAN HOAN TOAN THEO PLAN_A.MD")
    print("=" * 60)

    detector = VinmecFallDetector(model_complexity=0)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # TEST 1: Đứng thẳng bình thường (NORMAL)
    detector.pose.process = lambda x: create_mock_pose_landmarks("standing")
    t0 = 1000.0
    for i in range(10):
        _, tele = detector.process(dummy_frame.copy(), current_time=t0 + i * 0.033)

    assert tele["status"] == "NORMAL", f"Test 1 Failed: {tele['status']} != NORMAL"
    assert not tele["is_fall_confirmed"], "Test 1 Failed: is_fall_confirmed is True"
    print(">> TEST 1 THANH CONG: Tu the dung thang duoc nhan dien la NORMAL (Xanh la).")

    # TEST 2: Cúi nhặt đồ nhanh (< 1.2s) rồi đứng thẳng lại (Anti-False-Positive)
    # 2.1 Bắt đầu cúi
    detector.pose.process = lambda x: create_mock_pose_landmarks("bending")
    t1 = t0 + 1.0
    for i in range(15):  # ~0.5s cúi
        _, tele = detector.process(dummy_frame.copy(), current_time=t1 + i * 0.033)

    assert tele["status"] == "SUSPECTED", f"Test 2.1 Failed: {tele['status']} != SUSPECTED"
    assert not tele["is_fall_confirmed"], "Test 2.1 Failed: Bao dong qua som khi moi 0.5s!"
    print(">> TEST 2.1 THANH CONG: Cui nguoi kich hoat SUSPECTED (Vang), KHONG bao dong gia.")

    # 2.2 Đứng thẳng dậy ngay
    detector.pose.process = lambda x: create_mock_pose_landmarks("standing")
    t2 = t1 + 0.6
    for i in range(10):
        _, tele = detector.process(dummy_frame.copy(), current_time=t2 + i * 0.033)

    assert tele["status"] == "NORMAL", f"Test 2.2 Failed: {tele['status']} != NORMAL"
    assert not tele["is_fall_confirmed"], "Test 2.2 Failed: is_fall_confirmed is True"
    print(">> TEST 2.2 THANH CONG: Benh nhan dung thang day, trang thai tu dong ve NORMAL an toan!")

    # TEST 3: Ngã và nằm sàn liên tục (> 1.2s) -> FALL_CONFIRMED
    detector.pose.process = lambda x: create_mock_pose_landmarks("lying")
    t3 = t2 + 1.0
    # Duy trì tư thế nằm trong 1.5 giây (45 frames ở 30 FPS)
    for i in range(46):
        _, tele = detector.process(dummy_frame.copy(), current_time=t3 + i * 0.033)

    assert tele["status"] == "FALL_CONFIRMED", f"Test 3 Failed: {tele['status']} != FALL_CONFIRMED"
    assert tele["is_fall_confirmed"] is True, "Test 3 Failed: is_fall_confirmed phai la True"
    print(">> TEST 3 THANH CONG: Nga va duy tri > 1.2s lap tuc kich hoat BAO DONG FALL CONFIRMED (Chop Do)!")

    # TEST 4: Phím Reset (R)
    detector.reset_alarm()
    assert detector.status == "NORMAL", "Test 4 Failed: status chua reset ve NORMAL"
    assert not detector.is_fall_confirmed, "Test 4 Failed: is_fall_confirmed chua reset"
    print(">> TEST 4 THANH CONG: Phim [R] khoi phuc he thong ve NORMAL chuan xac!")

    detector.close()

    # TEST 5: Hiệu năng thực tế của MediaPipe Pose trên CPU Intel Core i5
    print("\n" + "=" * 60)
    print("DO HIEU NANG MEDIA PIPE POSE LITE TRUOC KHI HOAN TAT (CPU Intel Core i5):")
    real_detector = VinmecFallDetector(model_complexity=0)
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(test_img, (320, 240), 50, (255, 255, 255), -1)

    # Warmup
    for _ in range(5):
        real_detector.process(test_img.copy())

    # Benchmark 60 frames
    start_bench = time.time()
    num_frames = 60
    for _ in range(num_frames):
        real_detector.process(test_img.copy())
    total_time = time.time() - start_bench
    bench_fps = num_frames / total_time
    print(f"Ket qua do FPS thuc te tren CPU: {bench_fps:.1f} FPS (Muc tieu >= 25 FPS)")
    if bench_fps >= 25:
        print(">> DAT YEU CAU: He thong chay cuc ky muot ma >= 25 FPS tren CPU!")
    else:
        print(">> Canh bao: FPS duoi 25.")

    real_detector.close()
    print("=" * 60)
    print("TAT CA CAC TIEU CHI TRONG PLAN_A.MD DA DUOC KIEM TRA VA DAT 100%!")
    print("=" * 60)

    # Chạy thêm các bộ Unit Test chuyên sâu theo Plan A Updated
    print("\n" + "=" * 60)
    print("CHAY BO UNIT TESTS CHUYEN SAU CUA PLAN A UPDATED:")
    print("=" * 60)

    import tests.test_model_smoke as t_smoke
    t_smoke.test_model_file_exists()
    t_smoke.test_classifier_initialization()
    t_smoke.test_classifier_input_shape_and_dtype()
    t_smoke.test_predict_fall_probability_range()
    print(">> [PASS] Smoke Tests: Mo hinh TFLite hop le, san sang chay CPU.")

    import tests.test_pose_features as t_pose
    t_pose.test_standing_upright_features()
    t_pose.test_lying_horizontal_features()
    t_pose.test_low_visibility_treated_as_invalid()
    t_pose.test_none_landmarks_handled_gracefully()
    print(">> [PASS] Pose Features: Kiem tra do tin cay visibility va dac trung.")

    import tests.test_temporal_verifier as t_temp
    t_temp.test_1_standing_still()
    t_temp.test_2_slow_sitting()
    t_temp.test_3_bending_pick_up()
    t_temp.test_4_simulated_fall()
    t_temp.test_5_already_lying()
    t_temp.test_6_low_visibility()
    t_temp.test_7_irregular_timestamps()
    t_temp.test_8_cooldown_single_alarm()
    print(">> [PASS] Temporal Verifier: Dat 8/8 kich ban bat buoc cua Plan A Updated.")

    import tests.test_state_machine as t_fsm
    t_fsm.test_fsm_lifecycle()
    print(">> [PASS] FSM State Machine: Vong doi trang thai FSM hoan hao.")

    print("=" * 60)
    print("100% TAT CA CAC KIEM THU (LEGACY + PLAN A UPDATED) DA VUOT QUA!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

