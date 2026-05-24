from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from safety_monitor.models import ProximityState


class AlarmController:
    """UI 강조·선택적 소리·파일 로그·쿨다운."""

    def __init__(
        self,
        sound_enabled: bool,
        sound_file: str,
        log_path: str,
        cooldown_seconds: float,
    ) -> None:
        self.sound_enabled = sound_enabled
        self.sound_file = (sound_file or "").strip()
        self.log_path = Path(log_path) if log_path else None
        self.cooldown_seconds = float(cooldown_seconds)
        self._last_alarm_ts = 0.0
        self._player: Optional[QMediaPlayer] = None
        self._audio: Optional[QAudioOutput] = None
        self._log = logging.getLogger("safety_monitor.alarm")
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(self.log_path, encoding="utf-8")
            fh.setFormatter(
                logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
            )
            self._log.addHandler(fh)
            self._log.setLevel(logging.INFO)

    def configure(
        self,
        sound_enabled: bool,
        sound_file: str,
        log_path: str,
        cooldown_seconds: float,
    ) -> None:
        self.sound_enabled = sound_enabled
        self.sound_file = (sound_file or "").strip()
        self.cooldown_seconds = float(cooldown_seconds)
        if log_path and Path(log_path) != self.log_path:
            self.log_path = Path(log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def trigger(self, state: ProximityState, detail: str) -> None:
        if state == ProximityState.OUTSIDE:
            return
        now = time.time()
        if now - self._last_alarm_ts < self.cooldown_seconds:
            return
        self._last_alarm_ts = now
        level = "진입" if state == ProximityState.INSIDE else "접근"
        msg = f"{level}: {detail}"
        self._log.info(msg)
        if self.sound_enabled and self.sound_file and Path(self.sound_file).is_file():
            self._play_sound()

    def _play_sound(self) -> None:
        try:
            if self._player is None:
                self._player = QMediaPlayer()
                self._audio = QAudioOutput()
                self._player.setAudioOutput(self._audio)
            assert self._player is not None
            self._player.setSource(QUrl.fromLocalFile(str(Path(self.sound_file).resolve())))
            self._player.play()
        except Exception:  # noqa: BLE001
            pass
