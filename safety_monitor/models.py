from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple
import uuid


Point = Tuple[float, float]

# 신규 구역 기본 색 (순환)
ZONE_COLOR_PRESETS: Tuple[str, ...] = (
    "#5cb85c",
    "#5bc0de",
    "#f0ad4e",
    "#d9534f",
    "#9b59b6",
    "#3498db",
    "#1abc9c",
    "#e67e22",
    "#95a5a6",
    "#34495e",
)


def default_zone_color(zone_index: int) -> str:
    return ZONE_COLOR_PRESETS[zone_index % len(ZONE_COLOR_PRESETS)]


class ZoneShape(str, Enum):
    RECTANGLE = "rectangle"
    POLYGON = "polygon"


class ProximityState(str, Enum):
    OUTSIDE = "outside"
    APPROACHING = "approaching"
    INSIDE = "inside"


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int = 0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    def corners(self) -> List[Tuple[float, float]]:
        return [
            (self.x1, self.y1),
            (self.x2, self.y1),
            (self.x2, self.y2),
            (self.x1, self.y2),
        ]


@dataclass
class HazardZone:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "구역"
    enabled: bool = True
    shape: ZoneShape = ZoneShape.POLYGON
    points: List[Point] = field(default_factory=list)
    approach_margin_px: float = 40.0
    color_hex: str = "#6bce8f"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "shape": self.shape.value,
            "points": [list(p) for p in self.points],
            "approach_margin_px": self.approach_margin_px,
            "color": self.color_hex,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HazardZone":
        shape = ZoneShape(d.get("shape", "polygon"))
        pts = [tuple(p) for p in d.get("points", [])]
        ch = d.get("color") or d.get("color_hex")
        if not ch or not isinstance(ch, str):
            ch = "#6bce8f"
        return cls(
            id=d.get("id") or str(uuid.uuid4()),
            name=d.get("name", "구역"),
            enabled=bool(d.get("enabled", True)),
            shape=shape,
            points=pts,
            approach_margin_px=float(d.get("approach_margin_px", 40)),
            color_hex=str(ch),
        )


@dataclass
class FramePacket:
    """파이프라인을 통과하는 단일 프레임(BGR uint8)."""

    bgr: "object"  # numpy array
    capture_left: int
    capture_top: int
    capture_width: int
    capture_height: int
    timestamp: float


@dataclass
class RiskResult:
    detections: List[Detection]
    zone_states: dict  # zone_id -> list of (detection_index, ProximityState)
    worst_state: ProximityState
    highlight: bool
