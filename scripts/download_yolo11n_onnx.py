#!/usr/bin/env python3
"""Ultralytics YOLO11n 검출 ONNX를 models/ 에 받습니다 (공식 assets 릴리스)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    dest = root / "models" / "yolo11n.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        print(f"이미 있음: {dest}")
        return
    print(f"다운로드: {URL}\n -> {dest}")
    urllib.request.urlretrieve(URL, dest)
    print("완료.")


if __name__ == "__main__":
    main()
