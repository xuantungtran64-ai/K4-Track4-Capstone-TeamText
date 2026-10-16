"""
Smoke test cho mô hình TFLite Fall Detection:
Kiểm tra khả năng load, input/output tensor shapes, dtypes và phạm vi dự đoán fall_probability.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.fall_image_classifier import FallImageClassifier


def test_model_file_exists():
    assert os.path.exists(config.MODEL_PATH), f"File model khong ton tai tai: {config.MODEL_PATH}"


def test_classifier_initialization():
    classifier = FallImageClassifier(
        model_path=config.MODEL_PATH,
        num_threads=config.INFERENCE_THREADS,
    )
    assert classifier.is_loaded, "Classifier khong the load model TFLite"
    assert classifier.input_details is not None
    assert classifier.output_details is not None


def test_classifier_input_shape_and_dtype():
    classifier = FallImageClassifier(model_path=config.MODEL_PATH)
    inp = classifier.input_details[0]
    out = classifier.output_details[0]

    assert list(inp["shape"]) == [1, 224, 224, 3], f"Input shape khong dung: {inp['shape']}"
    assert inp["dtype"] == np.float32, f"Input dtype khong phai float32: {inp['dtype']}"
    assert list(out["shape"]) == [1, 1], f"Output shape khong dung: {out['shape']}"
    assert out["dtype"] == np.float32, f"Output dtype khong phai float32: {out['dtype']}"


def test_predict_fall_probability_range():
    classifier = FallImageClassifier(model_path=config.MODEL_PATH, max_fps=100.0)
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    p_fall = classifier.predict_fall_probability(dummy_frame)
    assert p_fall is not None, "Dự đoán trả về None"
    assert 0.0 <= p_fall <= 1.0, f"p_fall nam ngoai khoang [0, 1]: {p_fall}"


if __name__ == "__main__":
    test_model_file_exists()
    test_classifier_initialization()
    test_classifier_input_shape_and_dtype()
    test_predict_fall_probability_range()
    print("[PASS] Tat ca smoke tests cho Fall Model deu dat 100%!")
