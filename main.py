"""Application entry point for the ultrasonic signal analyzer."""

from __future__ import annotations

import sys


def main() -> int:
    """Start the desktop application."""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QFont
        from app.main_window import MainWindow
    except ImportError as exc:
        print(
            "GUI dependencies are not installed. Install them with "
            "`pip install -r requirements.txt` before running the GUI."
        )
        raise SystemExit(1) from exc

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 9))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
