"""
Script đo đạc hiệu năng CPU toàn diện (CPU Benchmark):
- MediaPipe Pose Lite đơn lẻ.
- CNN Fall Classifier đơn lẻ (1, 2, 4 threads).
- Toàn bộ Pipeline 2 chốt tích hợp Capture -> Pose -> CNN -> Verifier -> Overlay.
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.fall_detection_pipeline import FallDetectionPipeline
from src.fall_image_classifier import FallImageClassifier


def run_benchmark():
    print("=" * 65)
    print("VINMEC SMART WARD - BENCHMARK HIEU NANG CPU TOAN DIEN")
    print("=" * 65)

    test_img = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
    # Vẽ người giả định đơn giản
    cv2.circle(test_img, (320, 160), 40, (255, 255, 255), -1)
    cv2.line(test_img, (320, 200), (320, 360), (255, 255, 255), 10)
    cv2.line(test_img, (320, 240), (260, 300), (255, 255, 255), 6)
    cv2.line(test_img, (320, 240), (380, 300), (255, 255, 255), 6)

    # 1. Benchmark CNN TFLite với số luồng 1, 2, 4
    print("\n1. DO HIEU NANG CNN FALL CLASSIFIER (TFLite / LiteRT):")
    for th in [1, 2, 4]:
        classifier = FallImageClassifier(
            model_path=config.MODEL_PATH,
            num_threads=th,
            max_fps=100.0,  # Không throttle khi benchmark
            input_mode="full_frame",
        )
        if not classifier.is_loaded:
            print("   [BO QUA] Mo hinh TFLite chua duoc cai dat.")
            break

        # Warmup
        for _ in range(5):
            classifier.predict_fall_probability(test_img.copy(), current_time=time.time())

        latencies = []
        for _ in range(30):
            t0 = time.perf_counter()
            classifier.last_infer_time = 0.0 # Force inference
            classifier.predict_fall_probability(test_img.copy(), current_time=time.time())
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_l = np.mean(latencies)
        p95_l = np.percentile(latencies, 95)
        fps = 1000.0 / mean_l
        print(f"   Threads={th}: Mean={mean_l:.2f}ms | P95={p95_l:.2f}ms | Tuong duong {fps:.1f} FPS")

    # 2. Benchmark Pipeline Tích Hợp Đầy Đủ
    print("\n2. DO HIEU NANG TOAN BO PIPELINE 2 CHOT (Real-Time):")
    pipeline = FallDetectionPipeline(model_complexity=0)

    # Warmup
    for _ in range(5):
        pipeline.process(test_img.copy())

    N = 60
    t_start = time.perf_counter()
    for i in range(N):
        simulated_time = t_start + i * 0.033 # Mô phỏng 30 FPS video time
        pipeline.process(test_img.copy(), current_time=simulated_time, fps_display=30.0)
    total_time = time.perf_counter() - t_start
    real_fps = N / total_time
    avg_frame_ms = (total_time / N) * 1000.0

    print(f"   Tong so frames xu ly: {N}")
    print(f"   Tong thoi gian chay:  {total_time:.2f}s")
    print(f"   Thoi gian trung binh: {avg_frame_ms:.2f} ms / frame")
    print(f"   Toc do Pipeline dat:  {real_fps:.1f} FPS")

    if real_fps >= 25.0:
        print("\n>> KET LUAN: DAT YEU CAU HOAN TOAN (>= 25 FPS tren CPU)!")
    else:
        print(f"\n>> KET LUAN: Pipeline dat {real_fps:.1f} FPS (Muc tieu >= 25 FPS).")

    pipeline.close()
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()

