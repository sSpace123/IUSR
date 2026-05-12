"""Main window for the ultrasonic signal analyzer desktop application."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.channel_panel import ChannelPanel
from app.control_panel import ControlPanel
from app.import_dialog import ImportDialog
from app.ui_helpers import make_step_label
from app.widgets.signal_plot_widget import AnalysisDisplayWidget
from core.ai_assistant import suggest_analysis_parameters
from core.data_loader import load_signal_file
from core.export import export_signal_csv
from core.feature_extraction import compute_basic_features
from core.filtering import narrowband_filter
from core.signal_data import MultiChannelSignal
from core.spectrum_analysis import compute_fft
from core.wavelet_analysis import compute_cwt


class MainWindow(QMainWindow):
    """Research-friendly but simple desktop workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("超声信号分析软件")
        self.resize(1420, 860)
        self.setAcceptDrops(True)

        self.signal: MultiChannelSignal | None = None
        self.processed_signal: MultiChannelSignal | None = None
        self.current_path: Path | None = None
        self.feature_rows: list[dict[str, float | str]] = []
        self.last_auto_adjustment = ""

        self.channel_panel = ChannelPanel()
        self.control_panel = ControlPanel()
        self.feature_table = QTableWidget(0, 12)
        self.feature_table.setHorizontalHeaderLabels(
            ["通道", "峰值", "峰峰值", "RMS", "能量", "主频(Hz)", "峰值时间(s)", "到达时间(s)"]
        )
        self.feature_table.setHorizontalHeaderLabels(
            [
                "Channel",
                "Peak",
                "Peak-Peak",
                "RMS",
                "Energy",
                "Avg Power",
                "Envelope Energy",
                "Envelope Area",
                "Dominant Freq",
                "Peak Time",
                "TOF",
                "Envelope Peak",
            ]
        )
        self.display = AnalysisDisplayWidget(self.feature_table)
        self.step_labels = [
            make_step_label("1 导入数据", True),
            make_step_label("2 信号处理"),
            make_step_label("3 特征分析"),
            make_step_label("4 结果导出"),
        ]

        self._build_layout()
        self._connect_signals()
        self._apply_style()
        self._set_step(0)
        self.statusBar().showMessage("就绪：请导入 CSV/TXT 信号文件")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept dropped local files."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Load the first dropped file."""
        urls = event.mimeData().urls()
        if urls:
            self._load_path(Path(urls[0].toLocalFile()))

    def _build_layout(self) -> None:
        shell = QWidget()
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(12, 12, 12, 8)
        shell_layout.setSpacing(10)

        top = QFrame()
        top.setObjectName("topBar")
        top_layout = QHBoxLayout(top)
        title = QLabel("超声信号分析软件")
        title.setObjectName("appTitle")
        top_layout.addWidget(title)
        top_layout.addStretch(1)
        for index, step in enumerate(self.step_labels):
            top_layout.addWidget(step)
            if index < len(self.step_labels) - 1:
                arrow = QLabel("→")
                arrow.setObjectName("stepArrow")
                top_layout.addWidget(arrow)
        shell_layout.addWidget(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.channel_panel)
        splitter.addWidget(self.display)
        splitter.addWidget(self.control_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([250, 900, 290])
        shell_layout.addWidget(splitter, 1)

        self.setCentralWidget(shell)

    def _connect_signals(self) -> None:
        self.channel_panel.import_requested.connect(self._choose_file)
        self.channel_panel.visibility_changed.connect(self._refresh_time_plot)
        self.control_panel.display_changed.connect(self._refresh_time_plot)
        self.control_panel.fft_requested.connect(self._run_fft)
        self.control_panel.filter_requested.connect(self._run_filter)
        self.control_panel.features_requested.connect(self._run_features)
        self.control_panel.wavelet_requested.connect(self._run_wavelet)
        self.control_panel.export_signal_requested.connect(self._export_signal_csv)
        self.control_panel.export_features_requested.connect(self._export_features_csv)
        self.control_panel.export_image_requested.connect(self._export_current_image)
        self.control_panel.ai_adjust_requested.connect(self._run_ai_adjustment)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入信号数据",
            "",
            "Signal Files (*.csv *.txt *.xlsx *.xls *.npy *.npz *.mat);;All Files (*)",
        )
        if path:
            self._load_path(Path(path))

    def _load_path(self, path: Path) -> None:
        try:
            if path.suffix.lower() in {".csv", ".txt"}:
                dialog = ImportDialog(
                    path,
                    self.control_panel.sample_rate_input.value(),
                    self,
                    ai_config=self.control_panel.ai_config(),
                )
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                self.signal = load_signal_file(path, options=dialog.selected_options())
            else:
                self.signal = load_signal_file(
                    path, sample_rate=self.control_panel.sample_rate_input.value()
                )
        except Exception as exc:
            self._show_error("导入失败", _friendly_error(exc))
            return

        self.current_path = path
        self.processed_signal = None
        self.feature_rows = []
        self.last_auto_adjustment = ""
        self.control_panel.sample_rate_input.setValue(self.signal.sample_rate)
        self._auto_tune_filter_defaults()
        self._auto_tune_wavelet_defaults()
        self.channel_panel.update_signal(path, self.signal)
        self.feature_table.setRowCount(0)
        self._refresh_time_plot()
        self._set_step(1)
        self.control_panel.set_ready("数据已导入，可开始分析")
        self.statusBar().showMessage(
            f"已导入：{path.name} | 采样率 {self.signal.sample_rate:.6g} Hz | "
            f"通道 {len(self.signal.channels)}"
        )

    def _refresh_time_plot(self) -> None:
        if self.signal is None:
            self.display.show_empty()
            return
        visible = self._visible_data()
        if not visible:
            self.statusBar().showMessage("未选择可显示通道")
            return
        time, unit = self._display_time()
        self.display.plot_time_multi(
            time,
            visible,
            self.channel_panel.channel_colors(),
            normalize=self.control_panel.normalize_check.isChecked(),
            stacked=self.control_panel.stacked_check.isChecked(),
            grid=self.control_panel.grid_check.isChecked(),
            time_unit=unit,
        )

    def _run_filter(self) -> None:
        if not self._require_signal():
            return
        try:
            self._validate_filter()
            self.control_panel.set_busy(True, "正在进行窄带提取...")
            QApplication.processEvents()
            lowcut, highcut = self.control_panel.center_band()
            filtered = {
                name: narrowband_filter(
                    values,
                    self.control_panel.sample_rate_input.value(),
                    self.control_panel.center_input.value(),
                    self.control_panel.bandwidth_input.value(),
                    order=self.control_panel.filter_order_input.value(),
                    cycles=self.control_panel.filter_cycles_input.value(),
                )
                for name, values in self.signal.channels.items()
            }
            self.processed_signal = MultiChannelSignal(
                name=f"{self.signal.name}-filtered",
                time=self.signal.time,
                channels=filtered,
                sample_rate=self.control_panel.sample_rate_input.value(),
                metadata={**self.signal.metadata, "lowcut": lowcut, "highcut": highcut},
            )
        except Exception as exc:
            self.control_panel.set_ready("处理失败")
            self._show_error("处理失败", _friendly_error(exc))
            return

        ready_message = "处理完成，已生成窄带信号"
        if self.last_auto_adjustment:
            ready_message = f"{ready_message}；{self.last_auto_adjustment}"
        self.control_panel.set_ready(ready_message)
        self._set_step(2)
        self._refresh_time_plot()
        self.statusBar().showMessage(f"{ready_message}，可查看频谱或导出 CSV")

    def _run_fft(self) -> None:
        if not self._require_signal():
            return
        try:
            spectra = {}
            for name, values in self._visible_data().items():
                freqs, amplitudes = compute_fft(values, self.control_panel.sample_rate_input.value())
                dominant = float(freqs[int(np.argmax(amplitudes[1:]) + 1)]) if freqs.size > 1 else 0.0
                spectra[name] = (freqs, amplitudes, dominant)
            self.display.plot_spectrum_multi(
                spectra,
                self.channel_panel.channel_colors(),
                grid=self.control_panel.grid_check.isChecked(),
            )
        except Exception as exc:
            self._show_error("频谱分析失败", _friendly_error(exc))
            return
        self._set_step(2)
        self.statusBar().showMessage("频谱已生成，虚线标注各通道主频")

    def _run_features(self) -> None:
        if not self._require_signal():
            return
        try:
            rows = []
            for name, values in self._visible_data().items():
                features = compute_basic_features(
                    self.signal.time, values, self.control_panel.sample_rate_input.value()
                )
                rows.append({"channel": name, **features})
            self.feature_rows = rows
            self._fill_feature_table(rows)
        except Exception as exc:
            self._show_error("特征计算失败", _friendly_error(exc))
            return
        self._set_step(3)
        self.display.show_features()
        self.statusBar().showMessage("特征参数已计算完成")

    def _run_wavelet(self) -> None:
        if not self._require_signal():
            return
        visible = self._visible_data()
        if not visible:
            self._show_error("小波变换失败", "请至少选择一个通道。")
            return
        try:
            self._validate_wavelet()
            self.control_panel.set_busy(True, "正在生成小波图...")
            QApplication.processEvents()
            name, values = next(iter(visible.items()))
            freqs, coefficients = compute_cwt(
                values,
                self.control_panel.sample_rate_input.value(),
                self.control_panel.cwt_min_input.value(),
                self.control_panel.cwt_max_input.value(),
                self.control_panel.cwt_points_input.value(),
                self.control_panel.wavelet_combo.currentText(),
            )
            self.display.plot_wavelet(coefficients, self.signal.time, freqs)
        except Exception as exc:
            self.control_panel.set_ready("小波生成失败")
            self._show_error("小波生成失败", _friendly_error(exc))
            return
        self.control_panel.set_ready(f"小波图已生成：{name}")
        if self.last_auto_adjustment:
            self.control_panel.set_ready(f"小波图已生成：{name}；{self.last_auto_adjustment}")
        self._set_step(3)
        self.statusBar().showMessage(f"小波图已生成：{name}")

    def _run_ai_adjustment(self) -> None:
        if not self._require_signal():
            return
        assert self.signal is not None
        config = self.control_panel.ai_config()
        try:
            if config.enabled:
                if not config.api_key:
                    raise ValueError("请先填写 API Key，或关闭“启用大模型 API”。")
                suggestion = suggest_analysis_parameters(
                    self.signal.time, self.signal.channels, self.signal.sample_rate, config
                )
            else:
                suggestion = self._local_parameter_suggestion()
            message = self._apply_parameter_suggestion(suggestion)
        except Exception as exc:
            self._show_error("智能识别失败", _friendly_error(exc))
            return
        QMessageBox.information(self, "智能识别参数", message)
        self.control_panel.set_ready(message)
        self.statusBar().showMessage(message)

    def _export_signal_csv(self) -> None:
        if not self._require_signal():
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            signal = self.processed_signal or self.signal
            export_signal_csv(signal, path, self.channel_panel.visible_channels())
        except Exception as exc:
            self._show_error("导出失败", _friendly_error(exc))
            return
        self._set_step(4)
        self.statusBar().showMessage(f"导出成功：{path}")

    def _export_features_csv(self) -> None:
        if not self.feature_rows:
            self._show_error("导出失败", "请先点击“计算特征”。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出特征表 CSV", "", "CSV (*.csv)")
        if not path:
            return
        fieldnames = list(self.feature_rows[0].keys())
        with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.feature_rows)
        self._set_step(4)
        self.statusBar().showMessage(f"特征表导出成功：{path}")

    def _export_current_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "导出当前图像", "", "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if not path:
            return
        try:
            self.display.export_current_image(path)
        except Exception as exc:
            self._show_error("导出图像失败", _friendly_error(exc))
            return
        self._set_step(4)
        self.statusBar().showMessage(f"图像导出成功：{path}")

    def _local_parameter_suggestion(self) -> dict[str, float | int | str]:
        assert self.signal is not None
        dominant_values = []
        for values in self.signal.channels.values():
            freqs, amplitudes = compute_fft(values, self.signal.sample_rate)
            if freqs.size > 1:
                dominant_values.append(float(freqs[int(np.argmax(amplitudes[1:]) + 1)]))
        nyquist = self.signal.sample_rate / 2.0
        center = float(np.median(dominant_values)) if dominant_values else nyquist * 0.2
        center = min(max(center, nyquist * 0.02), nyquist * 0.8)
        bandwidth = min(max(center * 0.5, nyquist * 0.05), nyquist * 0.5)
        return {
            "center_frequency_hz": center,
            "bandwidth_hz": bandwidth,
            "filter_cycles": self.control_panel.filter_cycles_input.value(),
            "wavelet": "morl",
            "cwt_min_hz": max(center - bandwidth, nyquist * 0.01),
            "cwt_max_hz": min(center + bandwidth * 2.0, nyquist * 0.95),
            "cwt_points": self.control_panel.cwt_points_input.value(),
            "reason": "基于当前信号主频的本地自动建议。",
        }

    def _apply_parameter_suggestion(self, suggestion: dict) -> str:
        assert self.signal is not None
        nyquist = self.signal.sample_rate / 2.0
        center = _coerce_float(suggestion.get("center_frequency_hz"), self.control_panel.center_input.value())
        bandwidth = _coerce_float(suggestion.get("bandwidth_hz"), self.control_panel.bandwidth_input.value())
        cycles = int(_coerce_float(suggestion.get("filter_cycles"), self.control_panel.filter_cycles_input.value()))
        center = min(max(center, nyquist * 0.02), nyquist * 0.9)
        bandwidth = min(max(bandwidth, nyquist * 0.01), nyquist * 0.8)
        self.control_panel.center_input.setValue(center)
        self.control_panel.bandwidth_input.setValue(bandwidth)
        self.control_panel.filter_cycles_input.setValue(min(max(cycles, 1), 20))
        self._validate_filter()

        wavelet = str(suggestion.get("wavelet") or self.control_panel.wavelet_combo.currentText())
        index = self.control_panel.wavelet_combo.findText(wavelet)
        if index >= 0:
            self.control_panel.wavelet_combo.setCurrentIndex(index)

        cwt_min = _coerce_float(suggestion.get("cwt_min_hz"), self.control_panel.cwt_min_input.value())
        cwt_max = _coerce_float(suggestion.get("cwt_max_hz"), self.control_panel.cwt_max_input.value())
        self.control_panel.cwt_min_input.setValue(max(cwt_min, 1e-9))
        self.control_panel.cwt_max_input.setValue(max(cwt_max, 1e-9))
        self._validate_wavelet()
        cwt_points = int(_coerce_float(suggestion.get("cwt_points"), self.control_panel.cwt_points_input.value()))
        self.control_panel.cwt_points_input.setValue(min(max(cwt_points, 8), 512))

        reason = str(suggestion.get("reason") or "已根据当前数据自动调整参数。")
        return f"参数已调整：中心频率 {self.control_panel.center_input.value():.6g} Hz，带宽 {self.control_panel.bandwidth_input.value():.6g} Hz；{reason}"

    def _visible_data(self) -> dict[str, np.ndarray]:
        source = self.processed_signal or self.signal
        if source is None:
            return {}
        visible = self.channel_panel.visible_channels()
        return {name: source.channels[name] for name in visible if name in source.channels}

    def _raw_visible_data(self) -> dict[str, np.ndarray]:
        if self.signal is None:
            return {}
        visible = self.channel_panel.visible_channels()
        return {name: self.signal.channels[name] for name in visible if name in self.signal.channels}

    def _display_time(self) -> tuple[np.ndarray, str]:
        assert self.signal is not None
        unit = self.control_panel.time_unit_combo.currentText()
        scale = {"s": 1.0, "ms": 1e3, "us": 1e6}[unit]
        return self.signal.time * scale, unit

    def _validate_filter(self) -> None:
        fs = self.control_panel.sample_rate_input.value()
        lowcut, highcut = self.control_panel.center_band()
        nyquist = fs / 2.0
        self.last_auto_adjustment = ""
        if lowcut <= 0 or highcut >= nyquist:
            lowcut, highcut = self._auto_adjust_filter_band(fs)
            self.last_auto_adjustment = (
                f"已自动调整滤波范围为 {lowcut:.6g} Hz - {highcut:.6g} Hz"
            )
        self.control_panel.lowcut_input.setValue(lowcut)
        self.control_panel.highcut_input.setValue(highcut)

    def _auto_tune_filter_defaults(self) -> None:
        if self.signal is None:
            return
        self._auto_adjust_filter_band(self.signal.sample_rate)

    def _auto_tune_wavelet_defaults(self) -> None:
        if self.signal is None:
            return
        self._auto_adjust_wavelet_range(self.signal.sample_rate)

    def _auto_adjust_filter_band(self, fs: float) -> tuple[float, float]:
        nyquist = fs / 2.0
        margin = max(nyquist * 0.02, 1e-9)
        max_bandwidth = max(nyquist - 2.0 * margin, margin)
        bandwidth = min(self.control_panel.bandwidth_input.value(), max_bandwidth)
        bandwidth = max(bandwidth, min(max_bandwidth, nyquist * 0.1))

        min_center = bandwidth / 2.0 + margin
        max_center = nyquist - bandwidth / 2.0 - margin
        if max_center < min_center:
            bandwidth = max(nyquist * 0.5, margin)
            min_center = bandwidth / 2.0 + margin
            max_center = nyquist - bandwidth / 2.0 - margin

        center = min(max(self.control_panel.center_input.value(), min_center), max_center)
        self.control_panel.center_input.setValue(center)
        self.control_panel.bandwidth_input.setValue(bandwidth)
        return center - bandwidth / 2.0, center + bandwidth / 2.0

    def _validate_wavelet(self) -> None:
        fs = self.control_panel.sample_rate_input.value()
        f_min = self.control_panel.cwt_min_input.value()
        f_max = self.control_panel.cwt_max_input.value()
        nyquist = fs / 2.0
        self.last_auto_adjustment = ""
        if f_min <= 0 or f_max >= nyquist or f_min >= f_max:
            f_min, f_max = self._auto_adjust_wavelet_range(fs)
            self.last_auto_adjustment = (
                f"已自动调整小波频率范围为 {f_min:.6g} Hz - {f_max:.6g} Hz"
            )

    def _auto_adjust_wavelet_range(self, fs: float) -> tuple[float, float]:
        nyquist = fs / 2.0
        max_freq = max(nyquist * 0.95, 1e-9)
        min_freq = max(min(self.control_panel.cwt_min_input.value(), max_freq * 0.5), 1e-9)
        current_max = self.control_panel.cwt_max_input.value()
        if current_max <= min_freq or current_max >= nyquist:
            current_max = max_freq
        if min_freq >= current_max:
            min_freq = max(current_max * 0.05, 1e-9)
        self.control_panel.cwt_min_input.setValue(min_freq)
        self.control_panel.cwt_max_input.setValue(current_max)
        return min_freq, current_max

    def _require_signal(self) -> bool:
        if self.signal is not None:
            return True
        self._show_error("缺少数据", "请先导入 CSV/TXT 信号文件。")
        return False

    def _fill_feature_table(self, rows: list[dict[str, float | str]]) -> None:
        self.feature_table.setRowCount(len(rows))
        keys = [
            "channel",
            "peak",
            "peak_to_peak",
            "rms",
            "energy",
            "average_power",
            "envelope_energy",
            "envelope_area",
            "dominant_frequency",
            "peak_time",
            "tof",
            "envelope_peak",
        ]
        for row_index, row in enumerate(rows):
            for column_index, key in enumerate(keys):
                value = row[key]
                text = str(value) if isinstance(value, str) else f"{value:.8g}"
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.feature_table.setItem(row_index, column_index, item)
        self.feature_table.resizeColumnsToContents()

    def _set_step(self, active_index: int) -> None:
        for index, label in enumerate(self.step_labels):
            label.setProperty("activeStep", index == active_index)
            label.style().unpolish(label)
            label.style().polish(label)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)
        self.statusBar().showMessage(message)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
                font-size: 13px;
                color: #111827;
            }
            QMainWindow, QWidget#centerPanel {
                background: #F7F9FC;
            }
            QFrame#topBar, QWidget#sidePanel, QGroupBox, QTabWidget::pane {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
            }
            QFrame#topBar {
                padding: 6px;
            }
            QLabel#appTitle {
                font-size: 18px;
                font-weight: 700;
                color: #111827;
            }
            QLabel[activeStep="true"] {
                color: #2563EB;
                font-weight: 700;
            }
            QLabel[activeStep="false"], QLabel#stepArrow, QLabel#mutedLabel {
                color: #6B7280;
            }
            QLabel#emptyState {
                color: #6B7280;
                font-size: 16px;
                background: #FFFFFF;
                border: 1px dashed #CBD5E1;
                border-radius: 8px;
                padding: 32px;
            }
            QGroupBox {
                margin-top: 12px;
                padding: 12px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #111827;
            }
            QPushButton {
                border-radius: 7px;
                padding: 8px 10px;
                font-weight: 600;
            }
            QPushButton[variant="primary"] {
                background: #2563EB;
                color: white;
                border: 1px solid #2563EB;
            }
            QPushButton[variant="primary"]:hover {
                background: #1D4ED8;
            }
            QPushButton[variant="secondary"] {
                background: #FFFFFF;
                color: #2563EB;
                border: 1px solid #BFDBFE;
            }
            QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
                background: #FFFFFF;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 5px 7px;
            }
            QTableWidget {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                gridline-color: #E5E7EB;
            }
            QHeaderView::section {
                background: #F3F4F6;
                border: 0;
                padding: 7px;
                font-weight: 700;
            }
            QTabBar::tab {
                background: #EEF2FF;
                color: #374151;
                padding: 8px 14px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #2563EB;
                color: white;
            }
            QLabel#statusHint {
                color: #10B981;
                background: #ECFDF5;
                border: 1px solid #BBF7D0;
                border-radius: 7px;
                padding: 8px;
            }
            """
        )


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _friendly_error(error: Exception | str) -> str:
    text = str(error)
    if "sample_rate is required" in text:
        return "无法识别采样率，请在导入预览窗口中手动输入采样率。"
    if "Nyquist" in text or "highcut" in text:
        return "滤波上限频率不能超过 Nyquist 频率。"
    if "Unsupported file type" in text:
        return "当前文件格式暂不支持，请使用 CSV/TXT 或常见科学数据格式。"
    return text
