"""Left-side file information and channel management panel."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.style_manager import COLORS
from app.ui_helpers import format_duration, format_rate, make_primary_button, make_secondary_button
from app.widgets.signal_plot_widget import CHANNEL_COLORS
from core.signal_data import MultiChannelSignal


class LeftPanel(QWidget):
    """File information and channel visibility management."""

    import_requested = Signal()
    clear_requested = Signal()
    visibility_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidePanel")
        self.setMinimumWidth(220)
        self.setMaximumWidth(260)
        self._checks: dict[str, QCheckBox] = {}

        # Header
        self.import_button = make_primary_button("导入数据")
        self.import_button.setMinimumHeight(38)
        self.reimport_button = make_secondary_button("重新导入")
        self.clear_button = make_secondary_button("清除当前内容")

        # File info
        self.file_name = QLabel("—")
        self.file_name.setObjectName("infoValue")
        self.file_name.setWordWrap(True)
        self.sample_rate = QLabel("—")
        self.sample_rate.setObjectName("infoValue")
        self.sample_count = QLabel("—")
        self.sample_count.setObjectName("infoValue")
        self.duration = QLabel("—")
        self.duration.setObjectName("infoValue")
        self.channel_count = QLabel("—")
        self.channel_count.setObjectName("infoValue")

        # Channel list
        self.channel_layout = QVBoxLayout()
        self.channel_layout.setSpacing(2)
        self.channel_empty = QLabel("请先导入数据")
        self.channel_empty.setObjectName("infoLabel")
        self.channel_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.channel_layout.addWidget(self.channel_empty)

        self._build_layout()
        self._connect_signals()

    def visible_channels(self) -> list[str]:
        return [name for name, check in self._checks.items() if check.isChecked()]

    def channel_colors(self) -> dict[str, str]:
        return {
            name: CHANNEL_COLORS[index % len(CHANNEL_COLORS)]
            for index, name in enumerate(self._checks)
        }

    def update_signal(self, path: str | Path, signal: MultiChannelSignal) -> None:
        self.file_name.setText(Path(path).name)
        self.sample_rate.setText(format_rate(signal.sample_rate))
        self.sample_count.setText(f"{signal.time.size:,}")
        self.duration.setText(format_duration(float(signal.time[-1] - signal.time[0])))
        self.channel_count.setText(str(len(signal.channels)))
        self._set_channels(signal.channel_names[:8])

    def clear(self) -> None:
        self.file_name.setText("—")
        self.sample_rate.setText("—")
        self.sample_count.setText("—")
        self.duration.setText("—")
        self.channel_count.setText("—")
        self._set_channels([])

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Action buttons
        layout.addWidget(self.import_button)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addWidget(self.reimport_button, 1)
        btn_row.addWidget(self.clear_button, 1)
        layout.addLayout(btn_row)

        # File info card
        info_box = QGroupBox("文件信息")
        info_grid = QGridLayout(info_box)
        info_grid.setContentsMargins(10, 16, 10, 10)
        info_grid.setVerticalSpacing(6)
        info_grid.setHorizontalSpacing(10)
        self._add_info_row(info_grid, 0, "文件名", self.file_name)
        self._add_info_row(info_grid, 1, "采样率", self.sample_rate)
        self._add_info_row(info_grid, 2, "采样点数", self.sample_count)
        self._add_info_row(info_grid, 3, "时长", self.duration)
        self._add_info_row(info_grid, 4, "通道数", self.channel_count)
        layout.addWidget(info_box)

        # Channel management card
        channel_box = QGroupBox("通道管理")
        channel_box_layout = QVBoxLayout(channel_box)
        channel_box_layout.setContentsMargins(10, 16, 10, 10)
        channel_box_layout.addLayout(self.channel_layout)
        channel_box_layout.addStretch(1)
        layout.addWidget(channel_box, 1)

        layout.addStretch(1)

    def _add_info_row(self, grid: QGridLayout, row: int, label: str, value: QLabel) -> None:
        lbl = QLabel(label)
        lbl.setObjectName("infoLabel")
        grid.addWidget(lbl, row, 0)
        grid.addWidget(value, row, 1)

    def _connect_signals(self) -> None:
        self.import_button.clicked.connect(self.import_requested.emit)
        self.reimport_button.clicked.connect(self.import_requested.emit)
        self.clear_button.clicked.connect(self.clear_requested.emit)

    def _set_channels(self, names: list[str]) -> None:
        while self.channel_layout.count():
            item = self.channel_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._checks.clear()

        if not names:
            self.channel_empty = QLabel("请先导入数据")
            self.channel_empty.setObjectName("infoLabel")
            self.channel_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.channel_layout.addWidget(self.channel_empty)
            return

        for index, name in enumerate(names):
            row = QFrame()
            row.setObjectName("channelRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(6, 3, 6, 3)
            row_layout.setSpacing(8)

            color_dot = QLabel()
            color_dot.setFixedSize(12, 12)
            color = CHANNEL_COLORS[index % len(CHANNEL_COLORS)]
            color_dot.setStyleSheet(
                f"background:{color}; border-radius:6px; border:1px solid rgba(0,0,0,0.1);"
            )

            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda _state: self.visibility_changed.emit())
            self._checks[name] = checkbox

            row_layout.addWidget(color_dot)
            row_layout.addWidget(checkbox, 1)
            self.channel_layout.addWidget(row)
