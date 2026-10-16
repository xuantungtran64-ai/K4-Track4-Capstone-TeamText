"""
Script tải mô hình Fall Detection pretrained từ Hugging Face.
Tải một lần duy nhất vào thư mục models/, không tải lại khi chạy ứng dụng.
"""

import os
import urllib.request
import hashlib
import sys

# Đảm bảo đường dẫn import từ thư mục gốc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def download_model(url: str = config.MODEL_URL, dest: str = config.MODEL_PATH):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        size_bytes = os.path.getsize(dest)
        print(f"[OK] Mo hinh da ton tai tai: {dest} ({size_bytes} bytes). Khong can tai lai.")
        return dest

    print(f"[DOWNLOADING] Dang tai mo hinh tu: {url} -> {dest} ...")
    urllib.request.urlretrieve(url, dest)
    size_bytes = os.path.getsize(dest)
    print(f"[THANH CONG] Da tai xong mo hinh ({size_bytes} bytes).")

    # Tính MD5 checksum
    with open(dest, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    print(f"[CHECKSUM MD5] {md5}")
    return dest


if __name__ == "__main__":
    download_model()

