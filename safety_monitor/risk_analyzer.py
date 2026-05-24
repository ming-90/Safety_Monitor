from __future__ import annotations

from typing import List

from safety_monitor.models import Detection, HazardZone, ProximityState, RiskResult
from safety_monitor.zones import proximity_for_detection, worst_state


def analyze_frame(
    detections: List[Detection],
    zones: List[HazardZone],
    frame_shape: tuple,
) -> RiskResult:
    h, w = frame_shape[:2]
    zone_states: dict = {}
    flat: List[ProximityState] = []

    for z in zones:
        per_det: List[tuple] = []
        for i, det in enumerate(detections):
            st = proximity_for_detection(det, z, (h, w))
            per_det.append((i, st))
            flat.append(st)
        zone_states[z.id] = per_det

    wst = worst_state(flat)
    highlight = wst in (ProximityState.APPROACHING, ProximityState.INSIDE)
    return RiskResult(
        detections=detections,
        zone_states=zone_states,
        worst_state=wst,
        highlight=highlight,
    )
