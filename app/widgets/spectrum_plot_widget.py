"""Dedicated frequency-domain spectrum display widget."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from core.units import hz_to_frequency

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


class SpectrumPlotWidget(QWidget):
    """Frequency-domain spectrum with DC removal, dominant freq annotation, unit selection."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plots: list[pg.PlotItem] = []
        self._dominant_lines: list[pg.InfiniteLine] = []
        self._dominant_labels: list[pg.TextItem] = []

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.graphics)

    def clear(self) -> None:
        self.graphics.clear()
        self._plots = []
        self._dominant_lines = []
        self._dominant_labels = []

    def plot(
        self,
        spectra: dict[str, tuple[np.ndarray, np.ndarray, float]],
        colors: dict[str, str],
        ignore_dc: bool = True,
        grid: bool = True,
        db_scale: bool = False,
        freq_unit: str = "kHz",
    ) -> None:
        self.clear()
        if not spectra:
            return

        ref: pg.PlotItem | None = None
        count = len(spectra)

        for idx, (name, (freqs, amplitudes, dominant)) in enumerate(spectra.items()):
            if ignore_dc and freqs.size > 1:
                f = freqs[1:]
                a = amplitudes[1:]
            else:
                f = freqs
                a = amplitudes

            if db_scale:
                a = np.abs(a)
                ref_val = np.max(a) if a.size else 1.0
                a = 20 * np.log10(np.maximum(a, ref_val * 1e-12) / ref_val)

            if f.size > 20000:
                step = int(np.ceil(f.size / 20000))
                f, a = f[::step], a[::step]
            f_display = hz_to_frequency(f, freq_unit)

            plot = self.graphics.addPlot(row=idx, col=0, title=name)
            plot.showGrid(x=grid, y=grid, alpha=0.22)
            plot.setMenuEnabled(False)

            if ref is not None:
                plot.setXLink(ref)
            else:
                ref = plot

            color = colors.get(name, CHANNEL_COLORS[idx % len(CHANNEL_COLORS)])
            plot.plot(f_display, a, pen=pg.mkPen(color, width=1.2))

            # Dominant frequency (recalculate after DC exclusion)
            if ignore_dc and freqs.size > 1:
                dominant = float(freqs[1 + int(np.argmax(amplitudes[1:]))])

            if dominant > 0:
                dominant_display = hz_to_frequency(dominant, freq_unit)
                line = pg.InfiniteLine(
                    pos=dominant_display,
                    angle=90,
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                )
                plot.addItem(line)
                self._dominant_lines.append(line)

                # Annotate dominant frequency with unit
                label = pg.TextItem(
                    f"{dominant_display:.3f} {freq_unit}",
                    color=color,
                    anchor=(0, 1),
                )
                label.setPos(dominant_display, np.max(a) * 0.95 if a.size else 0)
                plot.addItem(label)
                self._dominant_labels.append(label)

            ylabel = "幅值 / dB" if db_scale else "幅值谱"
            plot.setLabel("left", ylabel)

            if idx == count - 1:
                plot.setLabel("bottom", f"频率 / {freq_unit}")

            plot.getViewBox().setMouseEnabled(x=True, y=True)
            self._plots.append(plot)

    def reset_view(self) -> None:
        for p in self._plots:
            p.autoRange()

    def export_image(self, path: str) -> None:
        pixmap = self.graphics.grab()
        if not pixmap.save(str(path)):
            raise ValueError(f"Failed to save spectrum image to {path}")
