"""Dedicated wavelet time-frequency display widget."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QVBoxLayout, QWidget

from core.units import auto_time_unit, hz_to_frequency, seconds_to_time


def _preferred_colormap(name: str = "viridis"):
    for candidate in (name, "turbo", "viridis"):
        try:
            return pg.colormap.get(candidate)
        except Exception:
            continue
    return pg.colormap.get("viridis")


class WaveletPlotWidget(QWidget):
    """CWT heatmap with auto-scaled axes, compact colorbar, and unit support."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")

        self._plot = self.graphics.addPlot(title="小波时频图")
        self.image = pg.ImageItem(axisOrder="row-major")
        self._plot.addItem(self.image)
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=False, y=False)

        cmap = _preferred_colormap("viridis")
        self.image.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        self.colorbar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=cmap, width=12)
        self.colorbar.setImageItem(self.image, insert_in=self._plot)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.graphics)

    def clear(self) -> None:
        self.image.setImage(np.zeros((1, 1)), autoLevels=False)

    def plot(
        self,
        coefficients: np.ndarray,
        time: np.ndarray,
        freqs: np.ndarray,
        freq_unit: str = "kHz",
        time_unit: str = "",
        colormap: str = "viridis",
    ) -> None:
        if coefficients.ndim != 2:
            raise ValueError("CWT 系数必须是二维矩阵。")

        magnitude = np.abs(coefficients)
        if magnitude.shape[0] != freqs.size:
            magnitude = magnitude.T
        if magnitude.shape[0] != freqs.size or magnitude.shape[1] != time.size:
            raise ValueError("CWT 矩阵形状与时间/频率轴不匹配。")

        finite = magnitude[np.isfinite(magnitude)]
        if finite.size == 0:
            raise ValueError("CWT 结果不包含有限值。")

        lower, upper = np.percentile(finite, [1, 99.5])
        if lower == upper:
            lower, upper = float(finite.min()), float(finite.max() or 1.0)

        display_time_unit = time_unit
        if not display_time_unit:
            dur = float(time[-1] - time[0]) if time.size > 1 else float(time[-1])
            display_time_unit = auto_time_unit(dur)
        time_display = seconds_to_time(time, display_time_unit)
        freqs_display = hz_to_frequency(freqs, freq_unit)

        self.image.setImage(magnitude, autoLevels=False)
        self.image.setLevels((float(lower), float(upper)))

        if time_display.size > 1 and freqs_display.size > 1:
            freq_step = float(np.median(np.diff(freqs_display)))
            y0 = max(0.0, float(freqs_display[0] - freq_step / 2.0))
            y1 = float(freqs_display[-1] + freq_step / 2.0)
            rect = QRectF(
                float(time_display[0]),
                y0,
                float(time_display[-1] - time_display[0]),
                y1 - y0,
            )
            self.image.setRect(rect)
            self._plot.setXRange(float(time_display[0]), float(time_display[-1]), padding=0.0)
            self._plot.setYRange(0.0, y1, padding=0.0)

        self._plot.setLabel("bottom", f"时间 / {display_time_unit}")

        # Frequency label
        self._plot.setLabel("left", f"频率 / {freq_unit}")

        self.colorbar.setLevels((float(lower), float(upper)))

        # Update colormap
        cmap = _preferred_colormap(colormap)
        self.image.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        try:
            self.colorbar.setColorMap(cmap)
        except Exception:
            pass

    def reset_view(self) -> None:
        self._plot.autoRange()

    def export_image(self, path: str) -> None:
        pixmap = self.graphics.grab()
        if not pixmap.save(str(path)):
            raise ValueError(f"Failed to save wavelet image to {path}")
