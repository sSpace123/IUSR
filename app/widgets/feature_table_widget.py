"""Feature parameter table widget."""

from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

FEATURE_COLUMNS = [
    ("通道", "channel"),
    ("峰值", "peak"),
    ("峰峰值", "peak_to_peak"),
    ("RMS", "rms"),
    ("能量", "energy"),
    ("平均功率", "average_power"),
    ("包络能量", "envelope_energy"),
    ("包络面积", "envelope_area"),
    ("主频", "dominant_frequency"),
    ("峰值时间", "peak_time"),
    ("到达时间", "tof"),
    ("包络峰值", "envelope_peak"),
]


class FeatureTableWidget(QWidget):
    """Table showing features per channel, with empty-state fallback."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.stack = QStackedWidget()

        # Empty state
        self.empty_label = QLabel("点击「计算特征」后显示结果")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setObjectName("infoLabel")

        # Table
        self.table = QTableWidget(0, len(FEATURE_COLUMNS))
        self.table.setHorizontalHeaderLabels([col[0] for col in FEATURE_COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.stack.addWidget(self.empty_label)
        self.stack.addWidget(self.table)
        self.stack.setCurrentWidget(self.empty_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)

    def set_features(self, rows: list[dict[str, float | str]]) -> None:
        if not rows:
            self.stack.setCurrentWidget(self.empty_label)
            return

        keys = [col[1] for col in FEATURE_COLUMNS]
        self.table.setRowCount(len(rows))
        for ri, row in enumerate(rows):
            for ci, key in enumerate(keys):
                val = row.get(key, "")
                text = str(val) if isinstance(val, str) else f"{val:.8g}"
                item = QTableWidgetItem(text)
                self.table.setItem(ri, ci, item)
        self.table.resizeColumnsToContents()
        self.stack.setCurrentWidget(self.table)

    def clear(self) -> None:
        self.table.setRowCount(0)
        self.stack.setCurrentWidget(self.empty_label)

    def export_csv(self, path: str | Path) -> None:
        """Export current table rows to CSV with Chinese column headers."""
        keys = [col[1] for col in FEATURE_COLUMNS]
        headers = [col[0] for col in FEATURE_COLUMNS]
        with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for ri in range(self.table.rowCount()):
                row = [self.table.item(ri, ci).text() if self.table.item(ri, ci) else ""
                       for ci in range(len(keys))]
                writer.writerow(row)

    def has_data(self) -> bool:
        return self.table.rowCount() > 0
