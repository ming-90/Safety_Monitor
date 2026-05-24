from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from safety_monitor.inference import Detector
from safety_monitor.models import Detection, FramePacket
from safety_monitor.result_emit import emit_frame_detection_pair


class InferenceThread(threading.Thread):
    """프레임을 받아 검출기를 실행하고 (프레임, 검출 결과)를 보냅니다."""

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        in_queue: "queue.Queue[Optional[FramePacket]]",
        out_queue: "queue.Queue[Optional[Tuple[FramePacket, List[Detection]]]]",
        stop_event: threading.Event,
        on_error: Optional[Callable[[str], None]] = None,
        inference_max_fps: Optional[float] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.model_cfg = model_cfg
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.on_error = on_error
        # 0 이하·None 이면 매 프레임 추론, 양수면 초당 그 횟수만 검출기 실행
        self._infer_interval: Optional[float] = None
        if inference_max_fps is not None and inference_max_fps > 0:
            self._infer_interval = 1.0 / float(inference_max_fps)
        self._last_infer_mono = -1e9
        self._last_dets: List[Detection] = []
        self.detector = Detector(
            model_path=str(model_cfg.get("path", "")),
            input_size=int(model_cfg.get("input_size", 640)),
            conf=float(model_cfg.get("confidence_threshold", 0.35)),
            nms=float(model_cfg.get("nms_threshold", 0.45)),
            class_ids_person=model_cfg.get("class_ids_person", [0]),
            use_demo=bool(model_cfg.get("use_demo_detector", True)),
        )

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                pkt = self.in_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if pkt is None:
                break
            try:
                now = time.monotonic()
                if self._infer_interval is None or (
                    now - self._last_infer_mono >= self._infer_interval
                ):
                    dets = self.detector.detect(pkt.bgr)
                    self._last_infer_mono = now
                else:
                    dets = self._last_dets
                self._last_dets = dets
                emit_frame_detection_pair(self.out_queue, pkt, dets)
            except Exception as e:  # noqa: BLE001
                if self.on_error:
                    self.on_error(f"추론 오류: {e}")
                self._last_dets = []
                emit_frame_detection_pair(self.out_queue, pkt, [])
