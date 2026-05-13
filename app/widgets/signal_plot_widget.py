"""Central analysis display with tabbed time/spectrum/wavelet/feature views."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QStackedWidget, QTabWidget, QVBoxLayout, QWidget

from app.widgets.feature_table_widget import FeatureTableWidget
from app.widgets.spectrum_plot_widget import SpectrumPlotWidget
from app.widgets.wavelet_plot_widget import WaveletPlotWidget
from core.units import seconds_to_time

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


def _downsample_for_display(x: np.ndarray, y: np.ndarray, max_points: int = 20000):
    if x.size <= max_points:
        return x, y
    step = int(np.ceil(x.size / max_points))
    return x[::step], y[::step]


def _normalize(values: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak == 0:
        return values
    return values / peak



class AnalysisDisplayWidget(QWidget):
    """Tabbed display for time-domain, spectrum, wavelet, and feature views."""

    import_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._time_plots: list[pg.PlotItem] = []

        self.tabs = QTabWidget()

        # --- Tab 0: Time domain ---
        self.time_stack = QStackedWidget()

        # Empty state
        self.empty_frame = QFrame()
        self.empty_frame.setObjectName("emptyStateFrame")
        empty_layout = QVBoxLayout(self.empty_frame)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_title = QLabel("导入数据以开始分析")
        empty_title.setObjectName("emptyStateTitle")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_hint = QLabel("拖拽 CSV/TXT 文件到窗口，或点击下方按钮\n支持单通道、多通道和最多 8 通道显示\n若无时间列，软件将提示输入采样率")
        empty_hint.setObjectName("emptyStateHint")
        empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        import_btn = QPushButton("导入 CSV/TXT")
        import_btn.setProperty("variant", "primary")
        import_btn.setMinimumHeight(40)
        import_btn.setMinimumWidth(200)
        import_btn.clicked.connect(self.import_requested.emit)
        empty_layout.addWidget(empty_title)
        empty_layout.addSpacing(8)
        empty_layout.addWidget(empty_hint)
        empty_layout.addSpacing(16)
        empty_layout.addWidget(import_btn, 0, Qt.AlignmentFlag.AlignCenter)

        self.time_widget = pg.GraphicsLayoutWidget()
        self.time_widget.setBackground("w")

        self.time_stack.addWidget(self.empty_frame)
        self.time_stack.addWidget(self.time_widget)

        # --- Tab 1: Spectrum ---
        self.spectrum_widget = SpectrumPlotWidget()

        # --- Tab 2: Wavelet ---
        self.wavelet_widget = WaveletPlotWidget()

        # --- Tab 3: Feature table ---
        self.feature_widget = FeatureTableWidget()

        self.tabs.addTab(self.time_stack, "时域信号")
        self.tabs.addTab(self.spectrum_widget, "频域信号")
        self.tabs.addTab(self.wavelet_widget, "小波变换")
        self.tabs.addTab(self.feature_widget, "特征参数")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

    # ── Public API (backward compatible) ──

    def show_empty(self) -> None:
        self.time_stack.setCurrentWidget(self.empty_frame)
        self.tabs.setCurrentWidget(self.time_stack)

    def show_time(self) -> None:
        self.time_stack.setCurrentWidget(self.time_widget)
        self.tabs.setCurrentWidget(self.time_stack)

    def show_spectrum(self) -> None:
        self.tabs.setCurrentWidget(self.spectrum_widget)

    def show_wavelet(self) -> None:
        self.tabs.setCurrentWidget(self.wavelet_widget)

    def show_features(self) -> None:
        self.tabs.setCurrentWidget(self.feature_widget)
        self.feature_widget.stack.setCurrentWidget(self.feature_widget.table)

    def clear_all(self) -> None:
        self.time_widget.clear()
        self._time_plots = []
        self.spectrum_widget.clear()
        self.wavelet_widget.clear()
        self.feature_widget.clear()
        self.show_empty()

    # ── Time domain ──

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
        self.time_widget.clear()
        self._time_plots = []

        ref: pg.PlotItem | None = None
        count = max(len(channels), 1)
        for idx, (name, values) in enumerate(channels.items()):
            plot = self.time_widget.addPlot(row=idx, col=0, title=name)
            plot.showGrid(x=grid, y=grid, alpha=0.22)
            plot.setMenuEnabled(False)
            if ref is not None:
                plot.setXLink(ref)
            else:
                ref = plot

            x, y = _downsample_for_display(time, values)
            y = _normalize(y) if normalize else y.copy()
            if stacked:
                y = y - np.median(y)

            color = colors.get(name, CHANNEL_COLORS[idx % len(CHANNEL_COLORS)])
            plot.plot(x, y, pen=pg.mkPen(color, width=1.2), name=name)
            plot.setLabel("left", "幅值")

            if idx == count - 1:
                plot.setLabel("bottom", f"时间 / {time_unit}")

            plot.getViewBox().setMouseEnabled(x=True, y=True)
            self._time_plots.append(plot)

        self.show_time()

    # ── Spectrum ──

    def plot_spectrum_multi(
        self,
        spectra: dict[str, tuple[np.ndarray, np.ndarray, float]],
        colors: dict[str, str],
        grid: bool = True,
        ignore_dc: bool = True,
        db_scale: bool = False,
        freq_unit: str = "kHz",
    ) -> None:
        self.spectrum_widget.plot(
            spectra, colors,
            ignore_dc=ignore_dc, grid=grid,
            db_scale=db_scale, freq_unit=freq_unit,
        )
        self.show_spectrum()

    # ── Wavelet ──

    def plot_wavelet(
        self,
        coefficients: np.ndarray,
        time: np.ndarray,
        freqs: np.ndarray,
        freq_unit: str = "kHz",
        time_unit: str = "",
        colormap: str = "viridis",
    ) -> None:
        self.wavelet_widget.plot(
            coefficients, time, freqs,
            freq_unit=freq_unit, time_unit=time_unit, colormap=colormap,
        )
        self.show_wavelet()

    # ── Narrowband result ──

    def plot_narrowband_result(
        self,
        packet_time: np.ndarray,
        packet_signal: np.ndarray,
        envelope: np.ndarray | None = None,
        original_time: np.ndarray | None = None,
        original_signal: np.ndarray | None = None,
        filtered_full: np.ndarray | None = None,
        show_envelope: bool = True,
        show_original: bool = False,
        show_filtered_full: bool = False,
        time_unit: str = "us",
        grid: bool = True,
    ) -> None:
        """Plot narrowband wave packet extraction result."""
        self.time_widget.clear()
        self._time_plots = []

        plot = self.time_widget.addPlot(row=0, col=0, title="窄带波包提取结果")
        plot.showGrid(x=grid, y=grid, alpha=0.22)
        plot.setMenuEnabled(False)

        packet_time_display = seconds_to_time(packet_time, time_unit)

        if show_envelope and envelope is not None:
            t_pkt, s_pkt = _downsample_for_display(packet_time_display, packet_signal)
            plot.plot(t_pkt, s_pkt, pen=pg.mkPen("#93C5FD", width=0.8), name="窄带载波")
            t_pkt2, env = _downsample_for_display(packet_time_display, envelope)
            plot.plot(t_pkt2, env, pen=pg.mkPen("#2563EB", width=1.8), name="Hilbert 包络")
        else:
            t_pkt, s_pkt = _downsample_for_display(packet_time_display, packet_signal)
            plot.plot(t_pkt, s_pkt, pen=pg.mkPen("#2563EB", width=1.5), name="窄带波包")

        # Original signal in gray
        if show_original and original_time is not None and original_signal is not None:
            to_, so_ = _downsample_for_display(seconds_to_time(original_time, time_unit), original_signal)
            plot.plot(to_, so_, pen=pg.mkPen("#9CA3AF", width=0.7), name="原始信号")

        # Full-length filtered signal
        if show_filtered_full and filtered_full is not None and original_time is not None:
            to2, ff = _downsample_for_display(seconds_to_time(original_time, time_unit), filtered_full)
            plot.plot(to2, ff, pen=pg.mkPen("#10B981", width=0.7), name="全长滤波信号")

        plot.setLabel("left", "归一化幅值" if np.max(np.abs(packet_signal)) <= 1.01 else "幅值")
        plot.setLabel("bottom", f"时间 / {time_unit}")
        plot.getViewBox().setMouseEnabled(x=True, y=True)
        self._time_plots.append(plot)
        self.show_time()

    # ── Features ──

    def set_features(self, rows: list[dict[str, float | str]]) -> None:
        self.feature_widget.set_features(rows)

    def feature_table(self) -> FeatureTableWidget:
        return self.feature_widget

    # ── View helpers ──

    def reset_view(self) -> None:
        current = self.tabs.currentWidget()
        if current is self.time_stack:
            for p in self._time_plots:
                p.autoRange()
        elif current is self.spectrum_widget:
            self.spectrum_widget.reset_view()
        elif current is self.wavelet_widget:
            self.wavelet_widget.reset_view()

    def export_current_image(self, path: str) -> None:
        current = self.tabs.currentWidget()
        if current is self.time_stack:
            pixmap = self.time_widget.grab()
        elif current is self.spectrum_widget:
            pixmap = self.spectrum_widget.graphics.grab()
        elif current is self.wavelet_widget:
            pixmap = self.wavelet_widget.graphics.grab()
        elif current is self.feature_widget:
            pixmap = self.feature_widget.grab()
        else:
            pixmap = self.grab()
        if not pixmap.save(str(path)):
            raise ValueError(f"Failed to save image to {path}")
