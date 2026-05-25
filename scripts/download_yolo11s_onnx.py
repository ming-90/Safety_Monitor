#!/usr/bin/env python3
"""Ultralytics YOLO11s 체크포인트를 받아 ONNX로 변환합니다."""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    pt_path = models_dir / "yolo11s.pt"
    onnx_path = models_dir / "yolo11s.onnx"
    models_dir.mkdir(parents=True, exist_ok=True)
    if onnx_path.is_file():
        print(f"이미 있음: {onnx_path}")
        return
    if not pt_path.is_file():
        print(f"다운로드: {URL}\n -> {pt_path}")
        urllib.request.urlretrieve(URL, pt_path)

    from ultralytics import YOLO

    print(f"ONNX 변환: {pt_path}\n -> {onnx_path}")
    model = YOLO(str(pt_path))
    exported = Path(model.export(format="onnx", imgsz=640, opset=12))
    if exported.resolve() != onnx_path.resolve():
        exported.replace(onnx_path)
    print("완료.")


if __name__ == "__main__":
    main()
