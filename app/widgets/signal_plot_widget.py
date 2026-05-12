"""Analysis display widgets built on pyqtgraph."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import QLabel, QStackedWidget, QTableWidget, QTabWidget, QVBoxLayout, QWidget

CHANNEL_COLORS = [
    "#2563EB",
    "#F97316",
    "#10B981",
    "#EF4444",
    "#8B5CF6",
    "#14B8A6",
    "#EC4899",
    "#64748B",
]


class AnalysisDisplayWidget(QWidget):
    """Central display for time-domain, spectrum, wavelet, and empty states."""

    def __init__(self, feature_table: QTableWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.feature_table = feature_table

        self.tabs = QTabWidget()
        self.time_stack = QStackedWidget()
        self.empty_label = QLabel(
            "拖拽 CSV/TXT 文件到此处，或点击“导入数据”\n支持单通道和最多 8 通道信号"
        )
        self.empty_label.setObjectName("emptyState")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.time_plot = pg.PlotWidget(title="时域信号")
        self.spectrum_plot = pg.PlotWidget(title="频域信号")
        self.wavelet_view = pg.ImageView()

        for plot in (self.time_plot, self.spectrum_plot):
            plot.showGrid(x=True, y=True, alpha=0.22)
            plot.setBackground("w")

        self.time_stack.addWidget(self.empty_label)
        self.time_stack.addWidget(self.time_plot)
        self.tabs.addTab(self.time_stack, "时域信号")
        self.tabs.addTab(self.spectrum_plot, "频域信号")
        self.tabs.addTab(self.wavelet_view, "小波变换")
        self.tabs.addTab(self.feature_table, "特征参数")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

    def show_empty(self) -> None:
        """Show empty state."""
        self.time_stack.setCurrentWidget(self.empty_label)
        self.tabs.setCurrentWidget(self.time_stack)

    def show_time(self) -> None:
        """Show time-domain page."""
        self.time_stack.setCurrentWidget(self.time_plot)
        self.tabs.setCurrentWidget(self.time_stack)

    def show_spectrum(self) -> None:
        """Show spectrum page."""
        self.tabs.setCurrentWidget(self.spectrum_plot)

    def show_wavelet(self) -> None:
        """Show wavelet page."""
        self.tabs.setCurrentWidget(self.wavelet_view)

    def show_features(self) -> None:
        """Show feature table page."""
        self.tabs.setCurrentWidget(self.feature_table)

    def plot_time_multi(
        self,
        time: np.ndarray,
        channels: dict[str, np.ndarray],
        colors: dict[str, str],
        normalize: bool = False,
        stacked: bool = False,
        grid: bool = True,
        time_unit: str = "s",
    ) -> None:
        """Plot visible channels in the time domain."""
        self.time_plot.clear()
        self.time_plot.showGrid(x=grid, y=grid, alpha=0.22)
        for index, (name, values) in enumerate(channels.items()):
            x, y = _downsample_for_display(time, values)
            y = _normalize(y) if normalize else y.copy()
            if stacked:
                y = y + index * 2.2
            self.time_plot.plot(x, y, pen=pg.mkPen(colors[name], width=1.35), name=name)
        self.time_plot.setLabel("bottom", "Time", units=time_unit)
        self.time_plot.setLabel("left", "Amplitude")
        self.time_plot.addLegend(offset=(8, 8))
        self.show_time()

    def plot_spectrum_multi(
        self,
        spectra: dict[str, tuple[np.ndarray, np.ndarray, float]],
        colors: dict[str, str],
        grid: bool = True,
    ) -> None:
        """Plot visible channel FFT spectra and mark dominant frequencies."""
        self.spectrum_plot.clear()
        self.spectrum_plot.showGrid(x=grid, y=grid, alpha=0.22)
        for name, (freqs, amplitudes, dominant) in spectra.items():
            x, y = _downsample_for_display(freqs, amplitudes)
            self.spectrum_plot.plot(x, y, pen=pg.mkPen(colors[name], width=1.35), name=name)
            marker = pg.InfiniteLine(
                pos=dominant,
                angle=90,
                pen=pg.mkPen(colors[name], width=1, style=Qt.PenStyle.DashLine),
            )
            self.spectrum_plot.addItem(marker)
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setLabel("left", "Amplitude")
        self.spectrum_plot.addLegend(offset=(8, 8))
        self.show_spectrum()

    def plot_wavelet(self, coefficients: np.ndarray, time: np.ndarray, freqs: np.ndarray) -> None:
        """Display a CWT coefficient matrix with a color scale."""
        magnitude = np.abs(coefficients)
        self.wavelet_view.setImage(magnitude.T, autoLevels=True)
        if time.size > 1 and freqs.size > 1:
            rect = QRectF(time[0], freqs[0], time[-1] - time[0], freqs[-1] - freqs[0])
            self.wavelet_view.imageItem.setRect(rect)
        self.show_wavelet()

    def reset_view(self) -> None:
        """Restore current plot view range."""
        current = self.tabs.currentWidget()
        if current is self.time_stack:
            current = self.time_plot
        if hasattr(current, "autoRange"):
            current.autoRange()

    def export_current_image(self, path: str) -> None:
        """Save the current display as an image."""
        pixmap = self.tabs.currentWidget().grab()
        if not pixmap.save(path):
            raise ValueError(f"Could not save image to {path}")


def _normalize(values: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak == 0:
        return values
    return values / peak


def _downsample_for_display(
    x: np.ndarray, y: np.ndarray, max_points: int = 20000
) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    step = int(np.ceil(x.size / max_points))
    return x[::step], y[::step]
