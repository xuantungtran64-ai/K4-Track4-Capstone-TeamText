"""
Script kiểm tra chi tiết kiến trúc và metadata của mô hình TFLite Fall Detection (Phase 1 Audit).
Kiểm tra input shape, dtype, output tensor, đối chiếu nhãn và đo độ trễ suy luận.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import ai_edge_litert.interpreter as litert
    Interpreter = litert.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter


def inspect_model(model_path: str = config.MODEL_PATH):
    print("=" * 65)
    print("AUDIT & KIEM TRA METADATA MO HINH TFLITE FALL DETECTION")
    print("=" * 65)
    if not os.path.exists(model_path):
        print(f"[LOI] File mo hinh khong ton tai tai: {model_path}")
        return

    print(f"File Path: {model_path}")
    print(f"File Size: {os.path.getsize(model_path)} bytes (~4.55 MB)")

    interpreter = Interpreter(model_path=model_path, num_threads=config.INFERENCE_THREADS)
    interpreter.allocate_tensors()

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    print("\n--- INPUT TENSORS ---")
    for i, inp in enumerate(inputs):
        print(f"Input {i}: Name='{inp['name']}', Shape={inp['shape']}, Dtype={inp['dtype']}, Quantization={inp['quantization']}")

    print("\n--- OUTPUT TENSORS ---")
    for i, out in enumerate(outputs):
        print(f"Output {i}: Name='{out['name']}', Shape={out['shape']}, Dtype={out['dtype']}, Quantization={out['quantization']}")

    # Đối chiếu giá trị đầu vào & kiểm tra output
    input_idx = inputs[0]["index"]
    output_idx = outputs[0]["index"]

    print("\n--- KIEM TRA SMOKE TEST & ANH XA NHAN ---")
    samples = [
        ("Zero array [0..255]", np.zeros((1, 224, 224, 3), dtype=np.float32)),
        ("White array [255.0]", np.full((1, 224, 224, 3), 255.0, dtype=np.float32)),
        ("Random array [0..255]", np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)),
    ]

    for name, arr in samples:
        interpreter.set_tensor(input_idx, arr)
        interpreter.invoke()
        raw_val = float(interpreter.get_tensor(output_idx)[0][0])
        p_fall = float(np.clip(1.0 - raw_val, 0.0, 1.0))
        print(f"{name:25} -> Raw Sigmoid: {raw_val:.4f} | P(fall) mapped: {p_fall:.4f}")

    # Đo độ trễ suy luận trên CPU
    print("\n--- BENCHMARK LATENCY TFLITE TREN CPU (NUM_THREADS=2) ---")
    dummy = np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
    # Warmup
    for _ in range(5):
        interpreter.set_tensor(input_idx, dummy)
        interpreter.invoke()

    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_idx, dummy)
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    print(f"Mean Latency: {np.mean(latencies):.2f} ms")
    print(f"P95 Latency:  {np.percentile(latencies, 95):.2f} ms")
    print(f"Min Latency:  {np.min(latencies):.2f} ms")
    print(f"Max Latency:  {np.max(latencies):.2f} ms")
    print("=" * 65)


if __name__ == "__main__":
    inspect_model()

