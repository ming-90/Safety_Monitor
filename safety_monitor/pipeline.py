from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from safety_monitor.capture_thread import CaptureThread
from safety_monitor.config import resolve_model_path
from safety_monitor.inference_thread import InferenceThread
from safety_monitor.models import Detection, FramePacket, HazardZone, RiskResult
from safety_monitor.result_emit import emit_frame_detection_pair
from safety_monitor.risk_analyzer import analyze_frame


class CaptureRelayThread(threading.Thread):
    """추론 없이 캡처 큐 → 결과 큐로 전달 (구역 설정 모드용)."""

    def __init__(
        self,
        in_queue: "queue.Queue[Optional[FramePacket]]",
        out_queue: "queue.Queue[Optional[Tuple[FramePacket, List[Detection]]]]",
        stop_event: threading.Event,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.on_error = on_error

    def run(self) -> None:
        try:
            self._loop()
        except Exception as e:  # noqa: BLE001
            if self.on_error:
                self.on_error(f"캡처 전달 오류: {e}")

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                pkt = self.in_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if pkt is None:
                break
            emit_frame_detection_pair(self.out_queue, pkt, [])


class MonitoringPipeline:
    """캡처→추론 큐; 위험 분석은 UI 타이머(소비 스레드)에서 수행."""

    def __init__(
        self,
        full_config: Dict[str, Any],
        on_error: Optional[Callable[[str], None]] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        self.full_config = full_config
        self.config_path = config_path
        qsize = int(full_config.get("pipeline", {}).get("queue_max_size", 2))
        self.capture_queue: "queue.Queue[Optional[FramePacket]]" = queue.Queue(
            maxsize=max(1, qsize)
        )
        self.result_queue: "queue.Queue[Optional[Tuple[FramePacket, List[Detection]]]]" = (
            queue.Queue(maxsize=max(1, qsize))
        )
        self.stop_event = threading.Event()
        self._capture: Optional[CaptureThread] = None
        self._infer: Optional[InferenceThread] = None
        self._relay: Optional[CaptureRelayThread] = None
        self._capture_only: bool = False
        self.on_error = on_error

    @property
    def capture_only(self) -> bool:
        return self._capture_only

    def update_config(self, full_config: Dict[str, Any]) -> None:
        self.full_config = full_config

    def start(self, capture_only: bool = False) -> None:
        self.stop_event.clear()
        self._capture_only = bool(capture_only)
        cap_cfg = self.full_config.get("capture", {})
        model_cfg = self.full_config.get("model", {})
        self._capture = CaptureThread(
            cap_cfg, self.capture_queue, self.stop_event, self.on_error
        )
        self._infer = None
        self._relay = None
        self._capture.start()
        if self._capture_only:
            self._relay = CaptureRelayThread(
                self.capture_queue,
                self.result_queue,
                self.stop_event,
                self.on_error,
            )
            self._relay.start()
        else:
            model_cfg = dict(model_cfg)
            mp = str(model_cfg.get("path", "")).strip()
            if mp:
                model_cfg["path"] = resolve_model_path(mp, self.config_path)
            pipe = self.full_config.get("pipeline", {})
            raw_fps = pipe.get("inference_max_fps", 3.0)
            try:
                inference_max_fps = float(raw_fps) if raw_fps is not None else None
            except (TypeError, ValueError):
                inference_max_fps = 3.0
            if inference_max_fps is not None and inference_max_fps <= 0:
                inference_max_fps = None
            self._infer = InferenceThread(
                model_cfg,
                self.capture_queue,
                self.result_queue,
                self.stop_event,
                self.on_error,
                inference_max_fps=inference_max_fps,
            )
            self._infer.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._capture is not None:
            self._capture.join(timeout=3.0)
        try:
            self.capture_queue.put_nowait(None)
        except queue.Full:
            try:
                _ = self.capture_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.capture_queue.put_nowait(None)
            except queue.Full:
                pass
        if self._infer is not None:
            self._infer.join(timeout=3.0)
        if self._relay is not None:
            self._relay.join(timeout=3.0)
        self._capture = None
        self._infer = None
        self._relay = None
        self._capture_only = False
        while not self.capture_queue.empty():
            try:
                self.capture_queue.get_nowait()
            except queue.Empty:
                break
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

    def poll_latest(
        self,
        zones: List[HazardZone],
    ) -> Optional[Tuple[FramePacket, RiskResult]]:
        """논블로킹: 큐를 비우며 최신 추론·위험 결과만 반환."""
        latest: Optional[Tuple[FramePacket, List[Detection]]] = None
        while True:
            try:
                item = self.result_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            latest = item
        if latest is None:
            return None
        pkt, dets = latest
        h, w = pkt.bgr.shape[:2]
        risk = analyze_frame(dets, zones, (h, w))
        return pkt, risk
