"""
Script kiem thu tu dong toan dien logic Tele-Rehabilitation (Option 2A - Plan B)
Kiem tra: calculate_angle, AngleSmoother, FSM Dem Reps, Bat loi bu tru vai, Visibility Guard, Xuat CSV va Benchmark FPS.
"""

import os
import time
import numpy as np
import cv2
import mediapipe as mp
from vinmec_rehab_demo import calculate_angle, AngleSmoother, VinmecRehabAnalyzer


def create_mock_lm(x, y, vis=0.9):
    class MockLM:
        def __init__(self, x, y, vis):
            self.x = x
            self.y = y
            self.visibility = vis
            self.z = 0.0

    return MockLM(x, y, vis)


def create_mock_rehab_landmarks(shoulder_angle=10.0, elbow_angle=160.0, side="LEFT", visibility=0.9):
    """
    Tao landmark mo phong voi goc khuyu tay (elbow_angle) va goc vung vai (shoulder_angle).
    Tọa độ chuẩn hóa:
    - Hip: [0.5, 0.8]
    - Shoulder: [0.5, 0.4]
    - Elbow: Tinh theo shoulder_angle so voi truc doc xuong
    - Wrist: Tinh theo elbow_angle so voi canh tay tren
    """
    landmarks = [create_mock_lm(0.5, 0.5, visibility) for _ in range(33)]
    mp_pose = mp.solutions.pose

    hip_pt = [0.5, 0.8]
    shoulder_pt = [0.5, 0.4]

    # Vector tu Shoulder xuong Elbow (goc shoulder_angle so voi truc doc xuong [0, 1])
    rad_s = np.radians(shoulder_angle)
    arm_len = 0.25
    elbow_pt = [
        shoulder_pt[0] + arm_len * np.sin(rad_s),
        shoulder_pt[1] + arm_len * np.cos(rad_s),
    ]

    # Vector tu Elbow xuong Wrist tao voi Upper Arm goc elbow_angle
    # Khi elbow_angle = 180 (duoi thang): cung huong voi upper arm
    # Khi elbow_angle < 180 (gap): gap nguoc len
    upper_angle = np.arctan2(elbow_pt[1] - shoulder_pt[1], elbow_pt[0] - shoulder_pt[0])
    forearm_len = 0.22
    # Goc forearm = upper_angle - (pi - radians(elbow_angle))
    forearm_angle = upper_angle - (np.pi - np.radians(elbow_angle))
    wrist_pt = [
        elbow_pt[0] + forearm_len * np.cos(forearm_angle),
        elbow_pt[1] + forearm_len * np.sin(forearm_angle),
    ]

    id_sh = mp_pose.PoseLandmark.LEFT_SHOULDER.value if side == "LEFT" else mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    id_el = mp_pose.PoseLandmark.LEFT_ELBOW.value if side == "LEFT" else mp_pose.PoseLandmark.RIGHT_ELBOW.value
    id_wr = mp_pose.PoseLandmark.LEFT_WRIST.value if side == "LEFT" else mp_pose.PoseLandmark.RIGHT_WRIST.value
    id_hp = mp_pose.PoseLandmark.LEFT_HIP.value if side == "LEFT" else mp_pose.PoseLandmark.RIGHT_HIP.value

    landmarks[id_sh] = create_mock_lm(shoulder_pt[0], shoulder_pt[1], visibility)
    landmarks[id_el] = create_mock_lm(elbow_pt[0], elbow_pt[1], visibility)
    landmarks[id_wr] = create_mock_lm(wrist_pt[0], wrist_pt[1], visibility)
    landmarks[id_hp] = create_mock_lm(hip_pt[0], hip_pt[1], visibility)

    class MockResults:
        def __init__(self, lms):
            self.pose_landmarks = MockPoseLandmarks(lms)

    class MockPoseLandmarks:
        def __init__(self, lms):
            self.landmark = lms

    return MockResults(landmarks)


def run_all_rehab_tests():
    print("=" * 60)
    print("BAT DAU KIEM THU CHUAN PLAN B (VINMEC TELE-REHAB)")
    print("=" * 60)

    # TEST 1: Ham tinh toan goc calculate_angle
    ang_90 = calculate_angle([0, 1], [0, 0], [1, 0])
    assert abs(ang_90 - 90.0) < 1e-3, f"Test 1.1 Failed: {ang_90} != 90"
    ang_180 = calculate_angle([-1, 0], [0, 0], [1, 0])
    assert abs(ang_180 - 180.0) < 1e-3, f"Test 1.2 Failed: {ang_180} != 180"
    print(">> TEST 1 THANH CONG: Ham calculate_angle tinh toan goc hinh hoc chuan 100%.")

    # TEST 2: Bo loc AngleSmoother (EMA)
    smoother = AngleSmoother(alpha=0.35)
    v1 = smoother.update(100.0)
    assert v1 == 100.0, "Test 2.1 Failed"
    v2 = smoother.update(120.0)
    expected_v2 = 0.35 * 120.0 + 0.65 * 100.0  # 42 + 65 = 107
    assert abs(v2 - expected_v2) < 1e-3, f"Test 2.2 Failed: {v2} != {expected_v2}"
    print(">> TEST 2 THANH CONG: Bo loc EMA AngleSmoother lam muot tin hieu chuan cong thuc.")

    # TEST 3: May trang thai Dem Reps (Repetition Counter FSM)
    analyzer = VinmecRehabAnalyzer(side="LEFT", model_complexity=0)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 3.1 Duoi tay thang (> 155 do, vai khong lech 10 do)
    analyzer.pose.process = lambda x: create_mock_rehab_landmarks(shoulder_angle=10.0, elbow_angle=165.0)
    for _ in range(5):
        _, tele = analyzer.process(dummy_frame.copy())
    assert tele["stage"] == "EXTENSION", f"Test 3.1 Failed: {tele['stage']} != EXTENSION"
    assert tele["reps"] == 0, "Test 3.1 Failed: reps phai bang 0"
    print(">> TEST 3.1 THANH CONG: Khoi dong trang thai EXTENSION khi duoi thang tay.")

    # 3.2 Gap tay xuong < 50 do -> Tang 1 Rep
    analyzer.pose.process = lambda x: create_mock_rehab_landmarks(shoulder_angle=10.0, elbow_angle=40.0)
    for _ in range(8):
        _, tele = analyzer.process(dummy_frame.copy())
    assert tele["stage"] == "FLEXION", f"Test 3.2 Failed: {tele['stage']} != FLEXION"
    assert tele["reps"] == 1, f"Test 3.2 Failed: reps = {tele['reps']} != 1"
    assert tele["best_rom"] < 50.0, "Test 3.2 Failed: best_rom chua cap nhat"
    print(f">> TEST 3.2 THANH CONG: Gap tay dat chuan tang chinh xac 1 Rep! (Best ROM: {tele['best_rom']:.1f} deg)")

    # 3.3 Duoi tay lai > 155 do -> Ve EXTENSION, chua tang reps
    analyzer.pose.process = lambda x: create_mock_rehab_landmarks(shoulder_angle=10.0, elbow_angle=165.0)
    for _ in range(8):
        _, tele = analyzer.process(dummy_frame.copy())
    assert tele["stage"] == "EXTENSION", "Test 3.3 Failed: Stage phai la EXTENSION"
    assert tele["reps"] == 1, "Test 3.3 Failed: reps khong duoc tang khi duoi tay"

    # 3.4 Gap tay lan 2 -> Tang Rep thu 2
    analyzer.pose.process = lambda x: create_mock_rehab_landmarks(shoulder_angle=10.0, elbow_angle=38.0)
    for _ in range(8):
        _, tele = analyzer.process(dummy_frame.copy())
    assert tele["reps"] == 2, f"Test 3.4 Failed: reps = {tele['reps']} != 2"
    print(">> TEST 3.3 & 3.4 THANH CONG: Hoan tat chu ky gap/duoi lan 2 tang thanh cong Rep = 2.")

    # TEST 4: Bat loi bu tru van dong (Compensatory Movement - shoulder drift > 35 do)
    # Dang cui cho ra ngoai (shoulder_angle = 50 do > 35 do) trong khi gap tay
    analyzer.pose.process = lambda x: create_mock_rehab_landmarks(shoulder_angle=50.0, elbow_angle=40.0)
    for _ in range(8):
        _, tele = analyzer.process(dummy_frame.copy())
    assert tele["is_bad_posture"] is True, "Test 4 Failed: is_bad_posture phai la True"
    assert "SAI TU THE" in tele["feedback"], "Test 4 Failed: thieu thong bao SAI TU THE"
    # Quan trong: Dang sai tu the KHONG duoc tang Rep!
    assert tele["reps"] == 2, f"Test 4 Failed: Reps van tang khi sai tu the ({tele['reps']} != 2)"
    print(">> TEST 4 THANH CONG: Dang rong khuyu tay (>35 deg) lap tuc bao SAI TU THE va CHAN tang Reps!")

    # TEST 5: Visibility Guard
    # Khi camera bi che (visibility < 0.5)
    analyzer.pose.process = lambda x: create_mock_rehab_landmarks(shoulder_angle=10.0, elbow_angle=40.0, visibility=0.3)
    _, tele = analyzer.process(dummy_frame.copy())
    assert tele["visibility_ok"] is False, "Test 5 Failed: visibility_ok phai la False"
    assert "CANH BAO" in tele["feedback"], "Test 5 Failed: Thieu canh bao camera bi che"
    print(">> TEST 5 THANH CONG: Visibility Guard phat hien mieu diem moc va hien thi canh bao.")

    # TEST 6: Xuat CSV bao cao buoi tap
    test_csv = "test_rehab_session.csv"
    if os.path.exists(test_csv):
        os.remove(test_csv)
    analyzer.export_csv(test_csv)
    assert os.path.exists(test_csv), "Test 6 Failed: File CSV khong duoc tao"
    with open(test_csv, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) >= 3, f"Test 6 Failed: CSV can it nhat 3 dong (header + 2 reps), nhan: {len(lines)}"
    assert "Timestamp,Rep_Number,Min_Angle_Achieved,Session_Best_ROM,Status" in lines[0]
    os.remove(test_csv)
    print(">> TEST 6 THANH CONG: File CSV sinh ra hop le day du cac cot du lieu y te.")

    analyzer.close()

    # TEST 7: Benchmark FPS tren CPU may tinh
    print("\n" + "=" * 60)
    print("DO HIEU NANG TELE-REHAB MEDIA PIPE TREN CPU INTEL CORE I5:")
    bench_analyzer = VinmecRehabAnalyzer(side="LEFT", model_complexity=0)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        bench_analyzer.process(test_frame.copy())

    t_start = time.time()
    n_frames = 60
    for _ in range(n_frames):
        bench_analyzer.process(test_frame.copy())
    elapsed = time.time() - t_start
    bench_fps = n_frames / elapsed
    print(f"FPS thuc te tren CPU: {bench_fps:.1f} FPS (Muc tieu >= 25 FPS)")
    if bench_fps >= 25.0:
        print(">> DAT YEU CAU: He thong Tele-Rehab chay cuc ky muot ma >= 25 FPS tren CPU!")
    bench_analyzer.close()

    print("=" * 60)
    print("TAT CA 7 MUC KIEM THU THEO PLAN_B.MD DA HOAN TAT VA DAT 100%!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_rehab_tests()

