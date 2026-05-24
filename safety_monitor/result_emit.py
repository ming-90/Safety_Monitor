from __future__ import annotations

import queue
from typing import List, Optional, Tuple

from safety_monitor.models import Detection, FramePacket


def emit_frame_detection_pair(
    out_queue: "queue.Queue[Optional[Tuple[FramePacket, List[Detection]]]]",
    pkt: FramePacket,
    dets: List[Detection],
) -> None:
    """최신 프레임 우선 정책으로 (프레임, 검출) 쌍을 결과 큐에 넣습니다."""
    item: Tuple[FramePacket, List[Detection]] = (pkt, dets)
    try:
        out_queue.put_nowait(item)
    except queue.Full:
        try:
            _ = out_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            out_queue.put_nowait(item)
        except queue.Full:
            pass
