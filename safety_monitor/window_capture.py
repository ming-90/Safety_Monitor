from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass(frozen=True)
class WindowSource:
    window_id: int
    owner_name: str
    title: str
    width: int
    height: int

    @property
    def label(self) -> str:
        title = f" - {self.title}" if self.title else ""
        return f"{self.owner_name}{title} ({self.width}x{self.height})"


def _quartz():
    if platform.system() != "Darwin":
        raise RuntimeError("창 캡처는 현재 macOS에서만 지원됩니다.")
    try:
        import Quartz  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "창 캡처에는 pyobjc-framework-Quartz 패키지가 필요합니다."
        ) from e
    return Quartz


def list_capture_windows() -> List[WindowSource]:
    quartz = _quartz()
    infos = quartz.CGWindowListCopyWindowInfo(
        quartz.kCGWindowListOptionOnScreenOnly,
        quartz.kCGNullWindowID,
    )
    windows: List[WindowSource] = []
    for info in infos:
        if int(info.get("kCGWindowLayer", 0)) != 0:
            continue
        bounds = info.get("kCGWindowBounds") or {}
        width = int(bounds.get("Width", 0))
        height = int(bounds.get("Height", 0))
        owner = str(info.get("kCGWindowOwnerName") or "").strip()
        title = str(info.get("kCGWindowName") or "").strip()
        window_id = int(info.get("kCGWindowNumber", 0))
        if not owner or window_id <= 0 or width < 80 or height < 80:
            continue
        windows.append(
            WindowSource(
                window_id=window_id,
                owner_name=owner,
                title=title,
                width=width,
                height=height,
            )
        )
    return windows


def capture_window_bgr(window_id: int) -> np.ndarray:
    quartz = _quartz()
    image = quartz.CGWindowListCreateImage(
        quartz.CGRectNull,
        quartz.kCGWindowListOptionIncludingWindow,
        int(window_id),
        quartz.kCGWindowImageBoundsIgnoreFraming,
    )
    if image is None:
        raise RuntimeError("선택한 창을 캡처할 수 없습니다.")

    width = int(quartz.CGImageGetWidth(image))
    height = int(quartz.CGImageGetHeight(image))
    bytes_per_row = int(quartz.CGImageGetBytesPerRow(image))
    provider = quartz.CGImageGetDataProvider(image)
    data = quartz.CGDataProviderCopyData(provider)
    raw = np.frombuffer(bytes(data), dtype=np.uint8)
    if width <= 0 or height <= 0 or bytes_per_row <= 0:
        raise RuntimeError("선택한 창의 캡처 크기가 올바르지 않습니다.")

    bgra = raw.reshape((height, bytes_per_row // 4, 4))[:, :width, :]
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
