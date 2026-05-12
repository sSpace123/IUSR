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
    """Central display for time-domain, spectrum, wavelet, and feature views."""

    def __init__(self, feature_table: QTableWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.feature_table = feature_table

        self.tabs = QTabWidget()
        self.time_stack = QStackedWidget()
        self.empty_label = QLabel(
            "拖拽 CSV/TXT 文件到此处，或点击“导入数据”\n支持单通道、多通道和最多 8 个文件/通道显示"
        )
        self.empty_label.setObjectName("emptyState")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.time_plot = pg.PlotWidget(title="时域信号")
        self.spectrum_plot = pg.PlotWidget(title="频域信号")
        self.wavelet_widget = pg.GraphicsLayoutWidget()
        self.wavelet_plot = self.wavelet_widget.addPlot(title="小波时频图")
        self.wavelet_image = pg.ImageItem(axisOrder="row-major")
        self.wavelet_plot.addItem(self.wavelet_image)
        self.wavelet_plot.setLabel("bottom", "Time", units="s")
        self.wavelet_plot.setLabel("left", "Frequency", units="Hz")
        self.wavelet_plot.showGrid(x=True, y=True, alpha=0.18)
        self.wavelet_plot.setLimits(xMin=0)
        self.wavelet_colorbar = _make_colorbar(self.wavelet_image, self.wavelet_plot)

        for plot in (self.time_plot, self.spectrum_plot):
            plot.showGrid(x=True, y=True, alpha=0.22)
            plot.setBackground("w")

        self.time_stack.addWidget(self.empty_label)
        self.time_stack.addWidget(self.time_plot)
        self.tabs.addTab(self.time_stack, "时域信号")
        self.tabs.addTab(self.spectrum_plot, "频域信号")
        self.tabs.addTab(self.wavelet_widget, "小波变换")
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
        self.tabs.setCurrentWidget(self.wavelet_widget)

    def show_features(self) -> None:
        """Show feature table page."""
        self.tabs.setCurrentWidget(self.feature_table)

    def clear_all(self) -> None:
        """Clear plots and return to the empty state."""
        self.time_plot.clear()
        self.spectrum_plot.clear()
        self.wavelet_image.setImage(np.zeros((1, 1)), autoLevels=False)
        self.show_empty()

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
        self.time_plot.addLegend(offset=(8, 8))
        self.time_plot.showGrid(x=grid, y=grid, alpha=0.22)
        for index, (name, values) in enumerate(channels.items()):
            x, y = _downsample_for_display(time, values)
            y = _normalize(y) if normalize else y.copy()
            if stacked:
                y = y + index * 2.2
            color = colors.get(name, CHANNEL_COLORS[index % len(CHANNEL_COLORS)])
            self.time_plot.plot(x, y, pen=pg.mkPen(color, width=1.35), name=name)
        self.time_plot.setLabel("bottom", "Time", units=time_unit)
        self.time_plot.setLabel("left", "Amplitude")
        self.show_time()

    def plot_spectrum_multi(
        self,
        spectra: dict[str, tuple[np.ndarray, np.ndarray, float]],
        colors: dict[str, str],
        grid: bool = True,
    ) -> None:
        """Plot visible channel FFT spectra and mark dominant frequencies."""
        self.spectrum_plot.clear()
        self.spectrum_plot.addLegend(offset=(8, 8))
        self.spectrum_plot.showGrid(x=grid, y=grid, alpha=0.22)
        for index, (name, (freqs, amplitudes, dominant)) in enumerate(spectra.items()):
            x, y = _downsample_for_display(freqs, amplitudes)
            color = colors.get(name, CHANNEL_COLORS[index % len(CHANNEL_COLORS)])
            self.spectrum_plot.plot(x, y, pen=pg.mkPen(color, width=1.35), name=name)
            self.spectrum_plot.addItem(
                pg.InfiniteLine(
                    pos=dominant,
                    angle=90,
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                )
            )
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setLabel("left", "Amplitude")
        self.show_spectrum()

    def plot_wavelet(self, coefficients: np.ndarray, time: np.ndarray, freqs: np.ndarray) -> None:
        """Display a CWT coefficient matrix with a scientific heatmap."""
        if coefficients.ndim != 2:
            raise ValueError("CWT coefficients must be a 2-D matrix.")
        magnitude = np.abs(coefficients)
        if magnitude.shape[0] != freqs.size:
            magnitude = magnitude.T
        if magnitude.shape[0] != freqs.size or magnitude.shape[1] != time.size:
            raise ValueError("CWT matrix shape does not match time/frequency axes.")

        finite = magnitude[np.isfinite(magnitude)]
        if finite.size == 0:
            raise ValueError("CWT result contains no finite values.")
        lower, upper = np.percentile(finite, [2, 98])
        if lower == upper:
            lower, upper = float(finite.min()), float(finite.max() or 1.0)

        self.wavelet_image.setImage(magnitude, autoLevels=False)
        self.wavelet_image.setLevels((float(lower), float(upper)))
        if time.size > 1 and freqs.size > 1:
            rect = QRectF(
                float(time[0]),
                float(freqs[0]),
                float(time[-1] - time[0]),
                float(freqs[-1] - freqs[0]),
            )
            self.wavelet_image.setRect(rect)
            self.wavelet_plot.setXRange(float(time[0]), float(time[-1]), padding=0.01)
            self.wavelet_plot.setYRange(float(freqs[0]), float(freqs[-1]), padding=0.01)
        _update_colorbar(self.wavelet_colorbar, float(lower), float(upper))
        self.show_wavelet()

    def reset_view(self) -> None:
        """Restore current plot view range."""
        current = self.tabs.currentWidget()
        if current is self.time_stack:
            current = self.time_plot
        if current is self.wavelet_widget:
            current = self.wavelet_plot
        if hasattr(current, "autoRange"):
            current.autoRange()

    def export_current_image(self, path: str) -> None:
        """Save the current display as an image."""
        pixmap = self.tabs.currentWidget().grab()
        if not pixmap.save(path):
            raise ValueError(f"Could not save image to {path}")


def _make_colorbar(image: pg.ImageItem, plot: pg.PlotItem):
    try:
        cmap = pg.colormap.get("viridis")
        image.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        colorbar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=cmap)
        colorbar.setImageItem(image, insert_in=plot)
        return colorbar
    except Exception:
        return None


def _update_colorbar(colorbar: object, lower: float, upper: float) -> None:
    if colorbar is None:
        return
    try:
        colorbar.setLevels((lower, upper))
    except Exception:
        pass


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
