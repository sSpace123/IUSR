"""Application entry point for the ultrasonic signal analyzer."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QSpinBox


class _ScrollWheelBlocker(QObject):
    """Global event filter that disables mouse-wheel value switching on
    QComboBox, QSpinBox, and QDoubleSpinBox so scrolling the panel does not
    unintentionally change parameter values."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel and isinstance(obj, (QComboBox, QSpinBox, QDoubleSpinBox)):
            return True  # eat the event
        return super().eventFilter(obj, event)


def main() -> int:
    """Start the desktop application."""
    try:
        from PySide6.QtGui import QFont
        from app.main_window import MainWindow
    except ImportError as exc:
        _write_startup_error(exc)
        print(
            "GUI dependencies are not installed. Install them with "
            "`pip install -r requirements.txt` before running the GUI.\n"
            f"Details: {exc}"
        )
        raise SystemExit(1) from exc

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 9))

    # Disable scroll-wheel value switching on combos and spin boxes
    scroll_blocker = _ScrollWheelBlocker()
    app.installEventFilter(scroll_blocker)

    window = MainWindow()
    window.show()
    return app.exec()


def _write_startup_error(error: BaseException) -> None:
    """Persist startup import failures for windowed PyInstaller builds."""
    try:
        base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()
        log_path = base / "UltrasonicSignalAnalyzer_startup_error.log"
        log_path.write_text(
            "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            encoding="utf-8",
        )
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
