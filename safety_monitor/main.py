from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from safety_monitor.ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    cfg = Path.cwd() / "config.json"
    win = MainWindow(config_path=cfg if cfg.is_file() else None)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
