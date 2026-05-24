from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QKeyEvent, QPainter, QPen, QColor, QPolygonF
from PySide6.QtWidgets import QWidget

from safety_monitor.models import (
    Detection,
    HazardZone,
    ProximityState,
    RiskResult,
    ZoneShape,
    default_zone_color,
)
from safety_monitor.zones import hit_test


class DrawMode(Enum):
    NONE = auto()
    POLY = auto()
    SELECT = auto()


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    return qimg.copy()


class VideoWidget(QWidget):
    """캡처 화면 표시, 위험 구역 오버레이, 그리기 도구."""

    zones_changed = Signal()
    selection_changed = Signal(object)
    zone_delete_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 360)
        self._bgr: Optional[np.ndarray] = None
        self.zones: List[HazardZone] = []
        self._risk: Optional[RiskResult] = None
        self.draw_mode = DrawMode.NONE
        self._poly_scratch: List[Tuple[float, float]] = []
        self._selected_id: Optional[str] = None
        self._drag_last: Optional[Tuple[float, float]] = None
        self._zone_drag_moved = False
        self._default_margin = 40.0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def selected_zone_id(self) -> Optional[str]:
        return self._selected_id

    def set_selected_zone_id(self, zid: Optional[str]) -> None:
        if self._selected_id != zid:
            self._selected_id = zid
            self.update()

    def set_frame(self, bgr: np.ndarray) -> None:
        self._bgr = bgr.copy()
        self.update()

    def set_zones(self, zones: List[HazardZone]) -> None:
        self.zones = zones
        self.update()

    def set_risk(self, risk: Optional[RiskResult]) -> None:
        self._risk = risk
        self.update()

    def set_default_margin(self, px: float) -> None:
        self._default_margin = float(px)

    def _image_size(self) -> Tuple[int, int]:
        if self._bgr is None:
            return (1280, 720)
        h, w = self._bgr.shape[:2]
        return w, h

    def _layout(self) -> Tuple[float, float, float, int, int]:
        """배율, 오프셋, 그려지는 이미지 크기(위젯 좌표)를 반환."""
        iw, ih = self._image_size()
        cw, ch = max(1, self.width()), max(1, self.height())
        scale = min(cw / iw, ch / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        ox = (cw - dw) * 0.5
        oy = (ch - dh) * 0.5
        return scale, ox, oy, dw, dh

    def widget_to_image(self, wx: float, wy: float) -> Tuple[float, float]:
        scale, ox, oy, _, _ = self._layout()
        if scale <= 0:
            return 0.0, 0.0
        ix = (wx - ox) / scale
        iy = (wy - oy) / scale
        return ix, iy

    def _finish_polygon_draw(self) -> None:
        """다각형 그리기: 3점 이상이면 구역 확정 후 목록 반영."""
        if self.draw_mode != DrawMode.POLY or len(self._poly_scratch) < 3:
            return
        z = HazardZone(
            name=f"구역 {len(self.zones) + 1}",
            shape=ZoneShape.POLYGON,
            points=list(self._poly_scratch),
            approach_margin_px=self._default_margin,
            color_hex=default_zone_color(len(self.zones)),
        )
        self.zones.append(z)
        self._poly_scratch.clear()
        self.zones_changed.emit()
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._bgr is None:
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        wx, wy = event.position().x(), event.position().y()
        ix, iy = self.widget_to_image(wx, wy)
        iw, ih = self._image_size()
        if not (0 <= ix < iw and 0 <= iy < ih):
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.draw_mode == DrawMode.POLY:
                self._poly_scratch.append((ix, iy))
            elif self.draw_mode == DrawMode.SELECT:
                self._zone_drag_moved = False
                for z in reversed(self.zones):
                    if hit_test(z, ix, iy):
                        self._selected_id = z.id
                        self._drag_last = (ix, iy)
                        self.selection_changed.emit(self._selected_id)
                        break
                else:
                    self._selected_id = None
                    self._drag_last = None
                    self.selection_changed.emit(None)
        elif event.button() == Qt.MouseButton.RightButton and self.draw_mode == DrawMode.POLY:
            if len(self._poly_scratch) >= 3:
                self._finish_polygon_draw()
            else:
                self._poly_scratch.clear()
                self.update()
        elif event.button() == Qt.MouseButton.RightButton and self.draw_mode == DrawMode.SELECT:
            for z in reversed(self.zones):
                if hit_test(z, ix, iy):
                    self.zone_delete_requested.emit(z.id)
                    break

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        wx, wy = event.position().x(), event.position().y()
        ix, iy = self.widget_to_image(wx, wy)
        if (
            self.draw_mode == DrawMode.SELECT
            and self._selected_id
            and self._drag_last
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            dx, dy = ix - self._drag_last[0], iy - self._drag_last[1]
            self._drag_last = (ix, iy)
            for z in self.zones:
                if z.id == self._selected_id:
                    z.points = [(px + dx, py + dy) for px, py in z.points]
                    break
            self._zone_drag_moved = True
            # 드래그 중에는 zones_changed 금지: 마우스 놓을 때 한 번만 알림(저장·카드 갱신).
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)
        if self._bgr is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._zone_drag_moved:
                self._zone_drag_moved = False
                self.zones_changed.emit()
            self._drag_last = None

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.draw_mode == DrawMode.POLY and len(self._poly_scratch) >= 3:
                self._finish_polygon_draw()
                event.accept()
                return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.draw_mode == DrawMode.POLY and self._poly_scratch:
                self._poly_scratch.clear()
                self.update()
                event.accept()
                return
            if self._selected_id is not None:
                self.zone_delete_requested.emit(self._selected_id)
                event.accept()
                return
        if event.key() == Qt.Key.Key_Escape and self.draw_mode == DrawMode.POLY and self._poly_scratch:
            self._poly_scratch.clear()
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 24))

        if self._bgr is None:
            painter.setPen(QPen(QColor(160, 160, 170)))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "「구역 설정 시작」 또는 「모니터링 시작」을 누르면 화면이 표시됩니다.",
            )
            return

        scale, ox, oy, dw, dh = self._layout()
        qimg = _bgr_to_qimage(self._bgr)
        painter.drawImage(QRectF(ox, oy, dw, dh), qimg)

        def _base_qcolor(z: HazardZone) -> QColor:
            c = QColor(z.color_hex)
            if not c.isValid():
                c = QColor("#6bce8f")
            return c

        def _worst_for_zone(zid: str) -> Optional[ProximityState]:
            if not self._risk or zid not in self._risk.zone_states:
                return None
            states = [s for _, s in self._risk.zone_states[zid]]
            if any(s == ProximityState.INSIDE for s in states):
                return ProximityState.INSIDE
            if any(s == ProximityState.APPROACHING for s in states):
                return ProximityState.APPROACHING
            return ProximityState.OUTSIDE

        def _mix(a: QColor, b: QColor, t: float) -> QColor:
            t = max(0.0, min(1.0, t))
            return QColor(
                int(a.red() * (1 - t) + b.red() * t),
                int(a.green() * (1 - t) + b.green() * t),
                int(a.blue() * (1 - t) + b.blue() * t),
            )

        def zone_stroke_for(z: HazardZone) -> QColor:
            base = _base_qcolor(z)
            if not z.enabled:
                return QColor(110, 110, 120, 200)
            w = _worst_for_zone(z.id)
            if w == ProximityState.INSIDE:
                return _mix(base, QColor(220, 40, 40), 0.55)
            if w == ProximityState.APPROACHING:
                return _mix(base, QColor(220, 150, 40), 0.45)
            d = QColor(base)
            d.setRed(max(0, int(d.red() * 0.72)))
            d.setGreen(max(0, int(d.green() * 0.72)))
            d.setBlue(max(0, int(d.blue() * 0.72)))
            d.setAlpha(235)
            return d

        def zone_fill_for(z: HazardZone) -> QColor:
            base = _base_qcolor(z)
            if not z.enabled:
                c = QColor(160, 165, 175)
                c.setAlpha(38)
                return c
            w = _worst_for_zone(z.id)
            fill = QColor(base)
            if w == ProximityState.INSIDE:
                fill = _mix(fill, QColor(255, 120, 120), 0.35)
            elif w == ProximityState.APPROACHING:
                fill = _mix(fill, QColor(255, 220, 120), 0.3)
            fill.setAlpha(62)
            return fill

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for z in self.zones:
            stroke = zone_stroke_for(z)
            fill = zone_fill_for(z)
            pen = QPen(stroke, 2)
            if not z.enabled:
                pen.setStyle(Qt.PenStyle.DashLine)
            if z.id == self._selected_id:
                pen = QPen(QColor(255, 200, 60), max(3, pen.width() + 2))
                if z.enabled:
                    pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            if z.shape == ZoneShape.RECTANGLE and len(z.points) >= 2:
                x0, y0 = z.points[0]
                x1, y1 = z.points[1]
                rx0, ry0 = ox + min(x0, x1) * scale, oy + min(y0, y1) * scale
                rw = abs(x1 - x0) * scale
                rh = abs(y1 - y0) * scale
                painter.fillRect(QRectF(rx0, ry0, rw, rh), fill)
                painter.drawRect(QRectF(rx0, ry0, rw, rh))
            elif len(z.points) >= 3:
                poly = QPolygonF([QPointF(ox + px * scale, oy + py * scale) for px, py in z.points])
                painter.setBrush(fill)
                painter.drawPolygon(poly)
                painter.setBrush(Qt.BrushStyle.NoBrush)

        if self._risk:
            inside_detection_ids = {
                det_idx
                for zone_states in self._risk.zone_states.values()
                for det_idx, state in zone_states
                if state == ProximityState.INSIDE
            }
            for idx, d in enumerate(self._risk.detections):
                if idx in inside_detection_ids:
                    painter.setPen(QPen(QColor(255, 40, 40), 3))
                else:
                    painter.setPen(QPen(QColor(0, 180, 255), 2))
                x0 = ox + d.x1 * scale
                y0 = oy + d.y1 * scale
                x1 = ox + d.x2 * scale
                y1 = oy + d.y2 * scale
                painter.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))

        if self._poly_scratch:
            pts = [QPointF(ox + px * scale, oy + py * scale) for px, py in self._poly_scratch]
            if len(self._poly_scratch) >= 3:
                poly = QPolygonF(pts)
                painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine))
                painter.setBrush(QColor(255, 255, 255, 28))
                painter.drawPolygon(poly)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
            for p in pts:
                painter.drawEllipse(p, 4, 4)

        if self._risk and self._risk.highlight:
            painter.setPen(QPen(QColor(255, 40, 40), 6))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.rect().adjusted(3, 3, -3, -3))

        painter.end()
