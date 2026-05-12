"""Left-side file information and channel management panel."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
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

from app.ui_helpers import format_duration, format_rate, make_primary_button, make_secondary_button
from app.widgets.signal_plot_widget import CHANNEL_COLORS
from core.signal_data import MultiChannelSignal


class ChannelPanel(QWidget):
    """Manage imports, file summary, and channel visibility."""

    import_requested = Signal()
    visibility_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidePanel")
        self._checks: dict[str, QCheckBox] = {}

        self.import_button = make_primary_button("导入数据")
        self.reimport_button = make_secondary_button("重新导入")
        self.file_name = QLabel("未导入文件")
        self.sample_rate = QLabel("-")
        self.sample_count = QLabel("-")
        self.duration = QLabel("-")
        self.channel_count = QLabel("-")
        self.channel_list = QVBoxLayout()

        self._build_layout()
        self.import_button.clicked.connect(self.import_requested.emit)
        self.reimport_button.clicked.connect(self.import_requested.emit)

    def visible_channels(self) -> list[str]:
        """Return currently checked channels."""
        return [name for name, check in self._checks.items() if check.isChecked()]

    def channel_colors(self) -> dict[str, str]:
        """Return stable colors for known channels."""
        return {
            name: CHANNEL_COLORS[index % len(CHANNEL_COLORS)]
            for index, name in enumerate(self._checks)
        }

    def update_signal(self, path: str | Path, signal: MultiChannelSignal) -> None:
        """Refresh file information and channel rows."""
        self.file_name.setText(Path(path).name)
        self.sample_rate.setText(format_rate(signal.sample_rate))
        self.sample_count.setText(str(signal.time.size))
        self.duration.setText(format_duration(float(signal.time[-1] - signal.time[0])))
        self.channel_count.setText(str(len(signal.channels)))
        self._set_channels(signal.channel_names[:8])

    def clear(self) -> None:
        """Reset the panel to its empty state."""
        self.file_name.setText("未导入文件")
        self.sample_rate.setText("-")
        self.sample_count.setText("-")
        self.duration.setText("-")
        self.channel_count.setText("-")
        self._set_channels([])

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(self.import_button)
        layout.addWidget(self.reimport_button)

        info_box = QGroupBox("文件信息")
        info_grid = QGridLayout(info_box)
        self._add_info_row(info_grid, 0, "文件", self.file_name)
        self._add_info_row(info_grid, 1, "采样率", self.sample_rate)
        self._add_info_row(info_grid, 2, "采样点", self.sample_count)
        self._add_info_row(info_grid, 3, "时长", self.duration)
        self._add_info_row(info_grid, 4, "通道数", self.channel_count)
        layout.addWidget(info_box)

        channel_box = QGroupBox("通道管理（最多 8 通道）")
        channel_layout = QVBoxLayout(channel_box)
        channel_layout.addLayout(self.channel_list)
        channel_layout.addStretch(1)
        layout.addWidget(channel_box, 1)
        layout.addStretch(1)

    def _add_info_row(self, layout: QGridLayout, row: int, label: str, value: QLabel) -> None:
        label_widget = QLabel(label)
        label_widget.setObjectName("mutedLabel")
        value.setWordWrap(True)
        layout.addWidget(label_widget, row, 0)
        layout.addWidget(value, row, 1)

    def _set_channels(self, names: list[str]) -> None:
        while self.channel_list.count():
            item = self.channel_list.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._checks.clear()

        if not names:
            empty = QLabel("导入数据后显示通道")
            empty.setObjectName("mutedLabel")
            self.channel_list.addWidget(empty)
            return

        for index, name in enumerate(names):
            row = QFrame()
            row.setObjectName("channelRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(6, 4, 6, 4)
            color = QLabel()
            color.setFixedSize(10, 10)
            color.setStyleSheet(
                f"background:{CHANNEL_COLORS[index % len(CHANNEL_COLORS)]};"
                "border-radius:5px;"
            )
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.visibility_changed.emit)
            self._checks[name] = checkbox
            row_layout.addWidget(color)
            row_layout.addWidget(checkbox, 1)
            self.channel_list.addWidget(row)

