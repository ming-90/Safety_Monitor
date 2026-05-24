from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from safety_monitor.models import HazardZone


def _parse_hex_color(hex_str: str) -> QColor:
    c = QColor(hex_str.strip())
    if not c.isValid():
        c = QColor("#6bce8f")
    return c


class ZoneCardFrame(QFrame):
    """단일 위험 구역 카드(이름·색·사용·여유·선택·삭제)."""

    select_toggled = Signal(str, bool)
    zone_changed = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, zone: HazardZone, parent=None) -> None:
        super().__init__(parent)
        self.zone = zone
        self._selected = False
        self.setObjectName("ZoneCardFrame")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self._build_ui()
        self._apply_selection_style()
        self._sync_from_zone()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._select_btn.blockSignals(True)
        self._select_btn.setChecked(selected)
        self._select_btn.blockSignals(False)
        self._apply_selection_style()

    def _emit_select_toggle(self, checked: bool) -> None:
        self.select_toggled.emit(self.zone.id, checked)

    def _apply_selection_style(self) -> None:
        if self._selected:
            self.setStyleSheet(
                "#ZoneCardFrame { background-color: #2f3440; border: 2px solid #f0b848; "
                "border-radius: 10px; }"
            )
        else:
            self.setStyleSheet(
                "#ZoneCardFrame { background-color: #262830; border: 1px solid #3d424d; "
                "border-radius: 10px; }"
            )

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        title = QLabel("위험 구역")
        title.setStyleSheet("color: #a8adb8; font-size: 11px;")
        outer.addWidget(title)

        row_name = QHBoxLayout()
        self._name = QLineEdit()
        self._name.setPlaceholderText("이름")
        self._name.editingFinished.connect(self._on_name_done)
        row_name.addWidget(self._name, stretch=1)

        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(36, 28)
        self._color_btn.setToolTip("구역 색 (지도 위 채우기·테두리)")
        self._color_btn.clicked.connect(self._pick_color)
        row_name.addWidget(self._color_btn)
        outer.addLayout(row_name)

        grid = QGridLayout()
        self._enabled = QCheckBox("사용")
        self._enabled.setToolTip("끄면 감시·알람에서 이 구역을 제외합니다.")
        self._enabled.toggled.connect(self._on_field_changed)
        grid.addWidget(self._enabled, 0, 0, 1, 2)

        margin_lay = QHBoxLayout()
        margin_lay.addWidget(QLabel("접근 여유(px)"))
        self._margin = QSpinBox()
        self._margin.setRange(0, 500)
        self._margin.setToolTip("경계 밖으로 확장한 접근 경고 띠 두께입니다.")
        self._margin.valueChanged.connect(self._on_field_changed)
        margin_lay.addWidget(self._margin)
        margin_lay.addStretch()
        grid.addLayout(margin_lay, 1, 0, 1, 2)

        outer.addLayout(grid)

        btn_row = QHBoxLayout()
        self._select_btn = QPushButton("선택")
        self._select_btn.setCheckable(True)
        self._select_btn.setToolTip("지도에서 이 구역 강조")
        self._select_btn.toggled.connect(self._emit_select_toggle)
        btn_row.addWidget(self._select_btn)

        del_btn = QPushButton("삭제")
        del_btn.setStyleSheet("color: #e07070;")
        del_btn.setToolTip("이 구역을 삭제합니다.")
        del_btn.clicked.connect(lambda: self.delete_requested.emit(self.zone.id))
        btn_row.addWidget(del_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

    def _sync_from_zone(self) -> None:
        self._name.blockSignals(True)
        self._enabled.blockSignals(True)
        self._margin.blockSignals(True)
        self._name.setText(self.zone.name)
        self._enabled.setChecked(self.zone.enabled)
        self._margin.setValue(int(self.zone.approach_margin_px))
        self._name.blockSignals(False)
        self._enabled.blockSignals(False)
        self._margin.blockSignals(False)
        self._update_color_swatch()

    def _update_color_swatch(self) -> None:
        c = _parse_hex_color(self.zone.color_hex)
        self._color_btn.setStyleSheet(
            f"QPushButton {{ background-color: {c.name()}; border: 1px solid #666; "
            f"border-radius: 6px; }}"
        )

    def _on_name_done(self) -> None:
        t = self._name.text().strip()
        if t:
            self.zone.name = t
        self.zone_changed.emit(self.zone.id)

    def _pick_color(self) -> None:
        cur = _parse_hex_color(self.zone.color_hex)
        dlg = QColorDialog.getColor(cur, self, "구역 색")
        if dlg.isValid():
            self.zone.color_hex = dlg.name()
            self._update_color_swatch()
            self.zone_changed.emit(self.zone.id)

    def _on_field_changed(self) -> None:
        self.zone.enabled = self._enabled.isChecked()
        self.zone.approach_margin_px = float(self._margin.value())
        self.zone_changed.emit(self.zone.id)
