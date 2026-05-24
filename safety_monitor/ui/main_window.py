from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from safety_monitor.alarm import AlarmController
from safety_monitor.config import (
    default_config_path,
    load_config,
    parse_zones,
    resolve_writable_path,
    save_config,
    zones_to_json,
)
from safety_monitor.models import HazardZone, ProximityState
from safety_monitor.pipeline import MonitoringPipeline
from safety_monitor.ui.video_widget import DrawMode, VideoWidget
from safety_monitor.ui.zone_cards import ZoneCardFrame
from safety_monitor.window_capture import list_capture_windows


class MainWindow(QMainWindow):
    def __init__(self, config_path: Optional[Path] = None) -> None:
        super().__init__()
        self.config_path = config_path
        self.cfg: Dict[str, Any] = load_config(config_path)
        self._ensure_defaults()
        self.zones: List[HazardZone] = parse_zones(self.cfg.get("hazard_zones", []))

        self.setWindowTitle("제조 현장 안전 모니터")
        self.resize(1280, 800)

        al = self.cfg.get("alarm", {})
        self.alarm = AlarmController(
            sound_enabled=bool(al.get("sound_enabled", True)),
            sound_file=str(al.get("sound_file", "")),
            log_path=resolve_writable_path(str(al.get("log_path", "logs/events.log"))),
            cooldown_seconds=float(al.get("cooldown_seconds", 2.0)),
        )

        self.video = VideoWidget()
        self.video.set_zones(self.zones)
        self.video.set_default_margin(
            float(self.cfg.get("alarm", {}).get("approach_margin_px", 40))
        )
        self.video.zones_changed.connect(self._on_zones_changed)
        self.video.selection_changed.connect(self._on_video_selection_changed)
        self.video.zone_delete_requested.connect(self._delete_zone_by_id)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self.video)
        self._splitter.addWidget(self._build_zone_side_panel())
        self._splitter.setStretchFactor(0, 4)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setSizes([920, 300])
        self.setCentralWidget(self._splitter)

        del_sc = QShortcut(QKeySequence.StandardKey.Delete, self)
        del_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        del_sc.activated.connect(self._shortcut_delete_zone)
        bs_sc = QShortcut(QKeySequence(Qt.Key.Key_Backspace), self)
        bs_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        bs_sc.activated.connect(self._shortcut_delete_zone)

        self._pipeline: Optional[MonitoringPipeline] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self._build_toolbar()
        self._zone_cards: Dict[str, ZoneCardFrame] = {}
        self._rebuild_zone_cards()

    def _ensure_defaults(self) -> None:
        self.cfg.setdefault("model", {})
        cap = self.cfg.setdefault("capture", {})
        cap.setdefault("mode", "monitor")
        cap.setdefault("monitor_index", 1)
        cap.setdefault("region", None)
        cap.setdefault("capture_max_fps", 15)
        cap.setdefault("window_id", None)
        cap.setdefault("window_owner", "")
        cap.setdefault("window_title", "")
        self.cfg.setdefault("hazard_zones", [])
        self.cfg.setdefault("alarm", {})
        self.cfg.setdefault("pipeline", {"queue_max_size": 2, "inference_max_fps": 3.0})

    def _build_toolbar(self) -> None:
        tb = QToolBar("메인")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._btn_start_setup = QPushButton("구역 설정 시작")
        self._btn_start_setup.setToolTip(
            "화면만 캡처합니다. 다각형·구역 편집 시 검출을 하지 않아 가볍습니다."
        )
        self._btn_start_monitor = QPushButton("모니터링 시작")
        self._btn_start_monitor.setToolTip(
            "캡처 후 사람 검출·위험 구역 판정을 수행합니다."
        )
        self._btn_stop = QPushButton("중지")
        self._btn_stop.setEnabled(False)
        self._btn_start_setup.clicked.connect(lambda: self._start(capture_only=True))
        self._btn_start_monitor.clicked.connect(lambda: self._start(capture_only=False))
        self._btn_stop.clicked.connect(self._stop)

        tb.addWidget(self._btn_start_setup)
        tb.addWidget(self._btn_start_monitor)
        tb.addWidget(self._btn_stop)
        tb.addSeparator()

        self._mon_spin = QSpinBox()
        self._mon_spin.setMinimum(1)
        self._mon_spin.setMaximum(16)
        self._mon_spin.setValue(int(self.cfg["capture"].get("monitor_index", 1)))
        self._mon_spin.valueChanged.connect(self._on_monitor_changed)
        tb.addWidget(QLabel("모니터 번호"))
        tb.addWidget(self._mon_spin)

        tb.addSeparator()
        self._window_combo = QComboBox()
        self._window_combo.setMinimumWidth(320)
        self._window_combo.currentIndexChanged.connect(self._on_window_source_changed)
        self._btn_refresh_windows = QPushButton("창 새로고침")
        self._btn_refresh_windows.clicked.connect(self._refresh_window_sources)
        tb.addWidget(QLabel("캡처 소스"))
        tb.addWidget(self._window_combo)
        tb.addWidget(self._btn_refresh_windows)
        self._refresh_window_sources()

        tb.addSeparator()
        for label, mode in [
            ("선택", DrawMode.SELECT),
            ("다각형 그리기", DrawMode.POLY),
        ]:
            b = QPushButton(label)
            b.setCheckable(True)

            def _mk(m: DrawMode, btn: QPushButton = b):
                def _on(checked: bool) -> None:
                    if checked:
                        self._uncheck_draw_buttons(except_btn=btn)
                        self.video.draw_mode = m
                    else:
                        self.video.draw_mode = DrawMode.NONE
                    self.video.update()

                return _on

            b.toggled.connect(_mk(mode, b))
            tb.addWidget(b)
            if mode == DrawMode.SELECT:
                self._btn_select = b
            else:
                self._btn_poly = b

        tb.addSeparator()
        save_btn = QPushButton("설정 저장")
        save_btn.clicked.connect(self._save_config)
        tb.addWidget(save_btn)

    def _uncheck_draw_buttons(self, except_btn: Optional[QPushButton] = None) -> None:
        for b in (getattr(self, "_btn_select", None), getattr(self, "_btn_poly", None)):
            if b is not None and b is not except_btn:
                b.blockSignals(True)
                b.setChecked(False)
                b.blockSignals(False)

    def _build_zone_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(420)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        list_title = QLabel("위험 구역")
        f = list_title.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 1)
        list_title.setFont(f)
        lay.addWidget(list_title)

        cards_scroll = QScrollArea()
        cards_scroll.setWidgetResizable(True)
        cards_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cards_scroll.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        self._zone_cards_host = QWidget()
        self._zone_cards_layout = QVBoxLayout(self._zone_cards_host)
        self._zone_cards_layout.setContentsMargins(0, 0, 4, 0)
        self._zone_cards_layout.setSpacing(10)
        cards_scroll.setWidget(self._zone_cards_host)
        lay.addWidget(cards_scroll, stretch=3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("알람 / 모델")
        gform = QFormLayout(grp)
        self._sound_on = QCheckBox()
        self._sound_on.setChecked(bool(self.cfg.get("alarm", {}).get("sound_enabled", True)))
        gform.addRow("소리", self._sound_on)

        self._sound_path = QLineEdit(str(self.cfg.get("alarm", {}).get("sound_file", "")))
        browse = QPushButton("찾아보기…")
        browse.clicked.connect(self._browse_sound)
        sp = QHBoxLayout()
        sp.addWidget(self._sound_path)
        sp.addWidget(browse)
        gform.addRow("소리 파일", sp)

        self._default_margin_alarm = QSpinBox()
        self._default_margin_alarm.setRange(0, 500)
        self._default_margin_alarm.setValue(
            int(self.cfg.get("alarm", {}).get("approach_margin_px", 40))
        )
        self._default_margin_alarm.valueChanged.connect(self._on_default_margin_changed)
        gform.addRow("신규 구역 기본 여유 (px)", self._default_margin_alarm)

        self._model_path = QLineEdit(str(self.cfg.get("model", {}).get("path", "")))
        mb = QPushButton("찾아보기…")
        mb.clicked.connect(self._browse_model)
        mp = QHBoxLayout()
        mp.addWidget(self._model_path)
        mp.addWidget(mb)
        gform.addRow("ONNX 모델", mp)

        self._demo_hog = QCheckBox("데모 HOG 검출기 사용 (ONNX 없음)")
        self._demo_hog.setChecked(bool(self.cfg.get("model", {}).get("use_demo_detector", True)))
        gform.addRow(self._demo_hog)

        inner_lay.addWidget(grp)
        inner_lay.addStretch()
        scroll.setWidget(inner)
        lay.addWidget(scroll, stretch=1)

        return panel

    def _rebuild_zone_cards(self) -> None:
        sel = self.video.selected_zone_id()
        while self._zone_cards_layout.count():
            item = self._zone_cards_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._zone_cards.clear()
        for z in self.zones:
            card = ZoneCardFrame(z)
            self._zone_cards[z.id] = card
            card.zone_changed.connect(self._on_card_zone_changed)
            card.delete_requested.connect(self._delete_zone_by_id)
            card.select_toggled.connect(self._on_card_select_toggled)
            self._zone_cards_layout.addWidget(card)
        self._zone_cards_layout.addStretch(1)
        if sel and sel in self._zone_cards:
            for cid, c in self._zone_cards.items():
                c.set_selected(cid == sel)
            self.video.set_selected_zone_id(sel)
        else:
            for c in self._zone_cards.values():
                c.set_selected(False)
            if sel:
                self.video.set_selected_zone_id(None)

    def _on_card_zone_changed(self, zid: str) -> None:
        if not any(z.id == zid for z in self.zones):
            return
        self.video.set_zones(self.zones)
        self.video.update()
        self._persist_hazard_zones()

    def _on_card_select_toggled(self, zid: str, checked: bool) -> None:
        if checked:
            for cid, c in self._zone_cards.items():
                c.set_selected(cid == zid)
            self.video.set_selected_zone_id(zid)
        else:
            if self.video.selected_zone_id() == zid:
                for c in self._zone_cards.values():
                    c.set_selected(False)
                self.video.set_selected_zone_id(None)
        self.video.update()

    def _on_video_selection_changed(self, zid: object) -> None:
        sid = str(zid) if zid else None
        for cid, card in self._zone_cards.items():
            card.set_selected(bool(sid) and cid == sid)
        self.video.update()

    def _delete_selected_zone(self) -> None:
        zid = self.video.selected_zone_id()
        if zid:
            self._delete_zone_by_id(str(zid))

    def _shortcut_delete_zone(self) -> None:
        w = QApplication.focusWidget()
        if isinstance(w, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
            return
        self._delete_selected_zone()

    def _delete_zone_by_id(self, zid: str) -> None:
        self.zones = [z for z in self.zones if z.id != zid]
        self.video.set_selected_zone_id(None)
        self._on_zones_changed()

    def _on_zones_changed(self) -> None:
        self._rebuild_zone_cards()
        self.video.set_zones(self.zones)
        self._persist_hazard_zones()

    def _persist_hazard_zones(self) -> None:
        """위험 구역만 설정 파일에 즉시 반영 (다음 실행 시 그대로 로드)."""
        self.cfg["hazard_zones"] = zones_to_json(self.zones)
        path = self.config_path or default_config_path()
        if self.config_path is None:
            self.config_path = path
        save_config(self.cfg, path)

    def _on_default_margin_changed(self, v: int) -> None:
        self.video.set_default_margin(float(v))

    def _on_monitor_changed(self, v: int) -> None:
        self.cfg["capture"]["monitor_index"] = int(v)
        self.cfg["capture"]["mode"] = "monitor"
        if hasattr(self, "_window_combo"):
            self._window_combo.setCurrentIndex(0)

    def _refresh_window_sources(self, *_args: object) -> None:
        raw_window_id = self.cfg.get("capture", {}).get("window_id")
        try:
            current_window_id = int(raw_window_id) if raw_window_id is not None else None
        except (TypeError, ValueError):
            current_window_id = None
        self._window_combo.blockSignals(True)
        self._window_combo.clear()
        self._window_combo.addItem("모니터 사용", {"mode": "monitor"})
        selected_idx = 0
        try:
            windows = list_capture_windows()
            for win in windows:
                data = {
                    "mode": "window",
                    "window_id": win.window_id,
                    "window_owner": win.owner_name,
                    "window_title": win.title,
                }
                self._window_combo.addItem(win.label, data)
                if (
                    self.cfg["capture"].get("mode") == "window"
                    and current_window_id is not None
                    and current_window_id == win.window_id
                ):
                    selected_idx = self._window_combo.count() - 1
        except Exception as e:  # noqa: BLE001
            self._window_combo.addItem(f"창 목록 오류: {e}", {"mode": "error"})
        self._window_combo.setCurrentIndex(selected_idx)
        self._window_combo.blockSignals(False)

    def _on_window_source_changed(self, _idx: int) -> None:
        data = self._window_combo.currentData() or {}
        if data.get("mode") == "window":
            self.cfg["capture"]["mode"] = "window"
            self.cfg["capture"]["window_id"] = int(data["window_id"])
            self.cfg["capture"]["window_owner"] = str(data.get("window_owner", ""))
            self.cfg["capture"]["window_title"] = str(data.get("window_title", ""))
        elif data.get("mode") == "monitor":
            self.cfg["capture"]["mode"] = "monitor"
            self.cfg["capture"]["monitor_index"] = self._mon_spin.value()

    def _browse_sound(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "소리 파일 선택", "", "WAV (*.wav)")
        if path:
            self._sound_path.setText(path)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "ONNX 모델 선택", "", "ONNX (*.onnx)")
        if path:
            self._model_path.setText(path)

    def _sync_settings_from_ui(self) -> None:
        source = self._window_combo.currentData() or {}
        if source.get("mode") == "window":
            self.cfg["capture"]["mode"] = "window"
            self.cfg["capture"]["window_id"] = int(source["window_id"])
            self.cfg["capture"]["window_owner"] = str(source.get("window_owner", ""))
            self.cfg["capture"]["window_title"] = str(source.get("window_title", ""))
        else:
            self.cfg["capture"]["mode"] = "monitor"
            self.cfg["capture"]["monitor_index"] = self._mon_spin.value()
        self.cfg["alarm"]["sound_enabled"] = self._sound_on.isChecked()
        self.cfg["alarm"]["sound_file"] = self._sound_path.text().strip()
        self.cfg["alarm"]["approach_margin_px"] = float(self._default_margin_alarm.value())
        self.cfg["model"]["path"] = self._model_path.text().strip()
        self.cfg["model"]["use_demo_detector"] = self._demo_hog.isChecked()
        self.cfg["hazard_zones"] = zones_to_json(self.zones)
        self.alarm.configure(
            sound_enabled=self.cfg["alarm"]["sound_enabled"],
            sound_file=self.cfg["alarm"]["sound_file"],
            log_path=resolve_writable_path(
                str(self.cfg["alarm"].get("log_path", "logs/events.log"))
            ),
            cooldown_seconds=float(self.cfg["alarm"].get("cooldown_seconds", 2.0)),
        )
        self.video.set_default_margin(float(self._default_margin_alarm.value()))

    def _save_config(self) -> None:
        self._sync_settings_from_ui()
        save_config(self.cfg, self.config_path)
        QMessageBox.information(
            self, "저장됨", "설정이 config.json에 저장되었습니다."
        )

    def _start(self, capture_only: bool = False) -> None:
        self._sync_settings_from_ui()
        if self._pipeline is not None:
            return
        self._pipeline = MonitoringPipeline(
            self.cfg, on_error=self._on_pipeline_error, config_path=self.config_path
        )
        self._pipeline.start(capture_only=capture_only)
        self._btn_start_setup.setEnabled(False)
        self._btn_start_monitor.setEnabled(False)
        self._btn_stop.setEnabled(True)
        if capture_only:
            self.setWindowTitle("제조 현장 안전 모니터 — 구역 설정 (검출 끔)")
        else:
            self.setWindowTitle("제조 현장 안전 모니터 — 모니터링")
        self._timer.start(30)

    def _stop(self) -> None:
        self._timer.stop()
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._btn_start_setup.setEnabled(True)
        self._btn_start_monitor.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self.setWindowTitle("제조 현장 안전 모니터")
        self.video.set_risk(None)

    def _on_pipeline_error(self, msg: str) -> None:
        QTimer.singleShot(0, lambda m=msg: QMessageBox.warning(self, "파이프라인", m))

    def _on_tick(self) -> None:
        if self._pipeline is None:
            return
        out = self._pipeline.poll_latest(self.zones)
        if out is None:
            return
        pkt, risk = out
        self.video.set_frame(pkt.bgr)
        self.video.set_risk(risk)
        if risk.worst_state != ProximityState.OUTSIDE:
            names = []
            for z in self.zones:
                if z.id not in risk.zone_states:
                    continue
                for _, st in risk.zone_states[z.id]:
                    if st != ProximityState.OUTSIDE:
                        names.append(z.name)
                        break
            self.alarm.trigger(risk.worst_state, ", ".join(names) or "위험구역")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop()
        super().closeEvent(event)
