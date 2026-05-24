from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import mss
import numpy as np

from safety_monitor.models import FramePacket
from safety_monitor.window_capture import capture_window_bgr


class CaptureThread(threading.Thread):
    """모니터/영역 캡처 후 최신 프레임을 큐에 넣습니다."""

    def __init__(
        self,
        capture_cfg: Dict[str, Any],
        out_queue: "queue.Queue[Optional[FramePacket]]",
        stop_event: threading.Event,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.capture_cfg = capture_cfg
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.on_error = on_error
        self._capture_interval = self._resolve_capture_interval()

    def _resolve_capture_interval(self) -> Optional[float]:
        raw = self.capture_cfg.get("capture_max_fps", self.capture_cfg.get("max_fps", 15))
        try:
            fps = float(raw)
        except (TypeError, ValueError):
            fps = 15.0
        if fps <= 0:
            return None
        return 1.0 / fps

    def _sleep_after_frame(self, frame_started_at: float) -> None:
        if self._capture_interval is None:
            return
        elapsed = time.perf_counter() - frame_started_at
        remaining = self._capture_interval - elapsed
        if remaining > 0:
            self.stop_event.wait(remaining)

    def _region_dict(self) -> Tuple[Dict[str, int], int, int]:
        mode = self.capture_cfg.get("mode", "monitor")
        with mss.mss() as sct:
            if mode == "region":
                reg = self.capture_cfg.get("region") or {}
                left = int(reg.get("left", 0))
                top = int(reg.get("top", 0))
                width = int(reg.get("width", 640))
                height = int(reg.get("height", 480))
                mon = {"left": left, "top": top, "width": width, "height": height}
                return mon, width, height
            idx = int(self.capture_cfg.get("monitor_index", 1))
            if idx < 1 or idx > len(sct.monitors) - 1:
                idx = 1
            mon = sct.monitors[idx]
            return mon, mon["width"], mon["height"]

    def run(self) -> None:
        try:
            self._run_loop()
        except Exception as e:  # noqa: BLE001
            if self.on_error:
                self.on_error(f"화면 캡처 오류: {e}")

    def _run_loop(self) -> None:
        if self.capture_cfg.get("mode") == "window":
            self._run_window_loop()
            return

        with mss.mss() as sct:
            mon, cw, ch = self._region_dict()
            while not self.stop_event.is_set():
                t0 = time.perf_counter()
                raw = np.array(sct.grab(mon))
                bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                pkt = FramePacket(
                    bgr=bgr,
                    capture_left=int(mon["left"]),
                    capture_top=int(mon["top"]),
                    capture_width=cw,
                    capture_height=ch,
                    timestamp=t0,
                )
                try:
                    self.out_queue.put_nowait(pkt)
                except queue.Full:
                    try:
                        _ = self.out_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.out_queue.put_nowait(pkt)
                    except queue.Full:
                        pass
                self._sleep_after_frame(t0)

    def _run_window_loop(self) -> None:
        window_id = int(self.capture_cfg.get("window_id") or 0)
        if window_id <= 0:
            raise RuntimeError("캡처할 창이 선택되지 않았습니다.")

        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            bgr = capture_window_bgr(window_id)
            ch, cw = bgr.shape[:2]
            pkt = FramePacket(
                bgr=bgr,
                capture_left=0,
                capture_top=0,
                capture_width=cw,
                capture_height=ch,
                timestamp=t0,
            )
            try:
                self.out_queue.put_nowait(pkt)
            except queue.Full:
                try:
                    _ = self.out_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.out_queue.put_nowait(pkt)
                except queue.Full:
                    pass
            self._sleep_after_frame(t0)
