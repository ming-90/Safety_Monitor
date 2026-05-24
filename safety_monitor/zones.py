from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from safety_monitor.models import Detection, HazardZone, ProximityState


def _zone_mask(shape_hw: Tuple[int, int], zone: HazardZone) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(zone.points) < 2:
        return mask
    pts = np.array(zone.points, dtype=np.int32).reshape((-1, 1, 2))
    if zone.shape.value == "rectangle" and len(zone.points) >= 2:
        x0, y0 = map(int, zone.points[0])
        x1, y1 = map(int, zone.points[1])
        cv2.rectangle(mask, (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y1, y1)), 255, -1)
    else:
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)
    return mask


def _bbox_mask(shape_hw: Tuple[int, int], det: Detection) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    x1 = int(max(0, min(w - 1, det.x1)))
    y1 = int(max(0, min(h - 1, det.y1)))
    x2 = int(max(0, min(w - 1, det.x2)))
    y2 = int(max(0, min(h - 1, det.y2)))
    if x2 <= x1 or y2 <= y1:
        return m
    m[y1:y2, x1:x2] = 255
    return m


def _dilate(mask: np.ndarray, margin_px: float) -> np.ndarray:
    if margin_px <= 0:
        return mask.copy()
    k = max(3, int(margin_px) * 2 + 1)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel)


def proximity_for_detection(
    det: Detection,
    zone: HazardZone,
    frame_shape: Tuple[int, int],
) -> ProximityState:
    """핵심 마스크와 팽창된 접근 띠로 검출 1건과 구역 1개의 관계를 분류."""
    if not zone.enabled or len(zone.points) < 2:
        return ProximityState.OUTSIDE

    h, w = frame_shape
    core = _zone_mask((h, w), zone)
    if not core.any():
        return ProximityState.OUTSIDE

    expanded = _dilate(core, zone.approach_margin_px)
    bbox = _bbox_mask((h, w), det)

    if cv2.countNonZero(cv2.bitwise_and(bbox, expanded)) == 0:
        return ProximityState.OUTSIDE

    cx, cy = int(det.cx), int(det.cy)
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    if core[cy, cx] > 0:
        return ProximityState.INSIDE

    return ProximityState.APPROACHING


def hit_test(zone: HazardZone, x: float, y: float) -> bool:
    """이미지 좌표 (x,y)가 구역 안인지(선택용)."""
    if len(zone.points) < 2:
        return False
    if zone.shape.value == "rectangle" and len(zone.points) >= 2:
        x0, y0 = zone.points[0]
        x1, y1 = zone.points[1]
        minx, maxx = min(x0, x1), max(x0, x1)
        miny, maxy = min(y0, y1), max(y0, y1)
        return minx <= x <= maxx and miny <= y <= maxy
    if len(zone.points) < 3:
        return False
    pts = np.array(zone.points, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


def worst_state(states: List[ProximityState]) -> ProximityState:
    order = {
        ProximityState.OUTSIDE: 0,
        ProximityState.APPROACHING: 1,
        ProximityState.INSIDE: 2,
    }
    if not states:
        return ProximityState.OUTSIDE
    return max(states, key=lambda s: order[s])
