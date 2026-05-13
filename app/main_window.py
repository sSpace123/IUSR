"""Main window for the ultrasonic signal analyzer."""

from __future__ import annotations

import csv
import json
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
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.import_dialog import ImportDialog
from app.left_panel import LeftPanel
from app.right_panel import RightPanel
from app.style_manager import stylesheet
from app.ui_helpers import format_rate, make_step_label
from app.widgets.signal_plot_widget import AnalysisDisplayWidget
from app.workers import CWTWorker, FeatureWorker, FilteringWorker
from core.ai_assistant import suggest_analysis_parameters
from core.data_loader import DataImportOptions, load_signal_file
from core.export import export_signal_csv
from core.feature_extraction import compute_basic_features, hilbert_envelope
from core.filtering import narrowband_filter
from core.signal_data import MultiChannelSignal
from core.spectrum_analysis import compute_fft, find_dominant_frequency
from core.units import (
    auto_time_unit,
    format_frequency,
    format_time,
    frequency_to_hz,
    hz_to_frequency,
    time_to_seconds,
)
from core.wavelet_analysis import estimate_cwt_cost, prepare_signal_for_cwt
from core.toneburst import generate_toneburst_preview


class MainWindow(QMainWindow):
    """Ultrasonic signal analyzer with three-column research workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("超声信号分析软件")
        self.resize(1420, 860)
        self.setMinimumSize(1280, 760)
        self.setAcceptDrops(True)

        self.signal: MultiChannelSignal | None = None
        self.processed_signal: MultiChannelSignal | None = None
        self.current_path: Path | None = None
        self.feature_rows: list[dict[str, float | str]] = []
        self.last_auto_adjustment = ""

        # Cached narrowband result for export
        self._narrowband_result: dict | None = None

        # Workers
        self._cwt_worker: CWTWorker | None = None
        self._filter_worker: FilteringWorker | None = None
        self._feature_worker: FeatureWorker | None = None

        # Panels
        self.left_panel = LeftPanel()
        self.right_panel = RightPanel()
        self.display = AnalysisDisplayWidget()

        # Top flow steps
        self.step_labels = [
            make_step_label("1 导入数据", True),
            make_step_label("2 信号处理"),
            make_step_label("3 特征分析"),
            make_step_label("4 结果导出"),
        ]

        # Status bar labels
        self._status_state = QLabel("就绪")
        self._status_file = QLabel("—")
        self._status_rate = QLabel("—")
        self._status_channels = QLabel("—")
        self._status_view = QLabel("—")

        self._build_layout()
        self._build_statusbar()
        self._connect_signals()
        self._apply_style()
        self._set_step(0)

    # ── Drag & drop ──

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self._load_paths(paths)

    # ── Layout ──

    def _build_layout(self) -> None:
        shell = QWidget()
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(12, 10, 12, 6)
        shell_layout.setSpacing(8)

        top = QFrame()
        top.setObjectName("topBar")
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(12, 8, 12, 8)
        title = QLabel("超声信号分析软件")
        title.setObjectName("appTitle")
        top_layout.addWidget(title)
        top_layout.addStretch(1)
        for i, step in enumerate(self.step_labels):
            step.setObjectName("stepLabel")
            top_layout.addWidget(step)
            if i < len(self.step_labels) - 1:
                arrow = QLabel("→")
                arrow.setObjectName("stepArrow")
                top_layout.addWidget(arrow)
        shell_layout.addWidget(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(3)
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.display)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([240, 800, 320])

        shell_layout.addWidget(splitter, 1)
        self.setCentralWidget(shell)

    def _build_statusbar(self) -> None:
        bar = QStatusBar()
        bar.setSizeGripEnabled(True)

        def _sep():
            lbl = QLabel("|")
            lbl.setStyleSheet("color:#D1D5DB;")
            bar.addPermanentWidget(lbl)

        bar.addWidget(QLabel("状态："))
        bar.addWidget(self._status_state)
        _sep()
        bar.addWidget(QLabel("文件："))
        bar.addWidget(self._status_file)
        _sep()
        bar.addWidget(QLabel("采样率："))
        bar.addWidget(self._status_rate)
        _sep()
        bar.addWidget(QLabel("通道："))
        bar.addWidget(self._status_channels)
        _sep()
        bar.addWidget(QLabel("视图："))
        bar.addWidget(self._status_view)
        self.setStatusBar(bar)

    def _update_status(self, state: str = "", file: str = "", rate: str = "",
                       channels: str = "", view: str = "") -> None:
        if state:
            self._status_state.setText(state)
        if file:
            self._status_file.setText(file)
        if rate:
            self._status_rate.setText(rate)
        if channels:
            self._status_channels.setText(channels)
        if view:
            self._status_view.setText(view)

    def _apply_style(self) -> None:
        self.setStyleSheet(stylesheet())

    # ── Signals ──

    def _connect_signals(self) -> None:
        self.left_panel.import_requested.connect(self._choose_file)
        self.left_panel.clear_requested.connect(self._clear_current)
        self.left_panel.visibility_changed.connect(self._refresh_time_plot)

        self.right_panel.display_changed.connect(self._refresh_time_plot)
        self.right_panel.fft_requested.connect(self._run_fft)
        self.right_panel.filter_requested.connect(self._run_narrowband_extraction)
        self.right_panel.features_requested.connect(self._run_features)
        self.right_panel.wavelet_requested.connect(self._run_wavelet)
        self.right_panel.wavelet_cancel_requested.connect(self._cancel_wavelet)
        self.right_panel.export_signal_requested.connect(self._export_narrowband_csv)
        self.right_panel.export_filtered_signal_requested.connect(self._export_filtered_csv)
        self.right_panel.export_envelope_requested.connect(self._export_envelope_csv)
        self.right_panel.export_features_requested.connect(self._export_features_csv)
        self.right_panel.export_image_requested.connect(self._export_current_image)
        self.right_panel.ai_adjust_requested.connect(self._run_ai_adjustment)
        self.display.import_requested.connect(self._choose_file)

    # ── File loading ──

    def _choose_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "导入信号文件",
            "",
            "Signal Files (*.csv *.txt *.xlsx *.xls *.npy *.npz *.mat);;All Files (*)",
        )
        if paths:
            self._load_paths([Path(p) for p in paths])

    def _load_paths(self, paths: list[Path]) -> None:
        try:
            signal = self._load_single_path(paths[0]) if len(paths) == 1 else self._load_multi_paths(paths)
        except Exception as exc:
            if isinstance(exc, _UserCancelled):
                return
            self._show_error("导入失败", _friendly_error(exc))
            return
        self._accept_loaded_signal(signal, paths[0], len(paths))

    def _load_single_path(self, path: Path) -> MultiChannelSignal:
        if path.suffix.lower() in {".csv", ".txt"}:
            options = self._ask_import_options(path)
            if options is None:
                raise _UserCancelled()
            return load_signal_file(path, options=options)
        fs = frequency_to_hz(
            self.right_panel.sample_rate_input.value(),
            self.right_panel.sample_rate_unit.currentText(),
        )
        return load_signal_file(path, sample_rate=fs)

    def _load_multi_paths(self, paths: list[Path]) -> MultiChannelSignal:
        options: DataImportOptions | None = None
        if paths[0].suffix.lower() in {".csv", ".txt"}:
            options = self._ask_import_options(paths[0])
            if options is None:
                raise _UserCancelled()

        signals: list[MultiChannelSignal] = []
        fs = frequency_to_hz(
            self.right_panel.sample_rate_input.value(),
            self.right_panel.sample_rate_unit.currentText(),
        )
        for path in paths:
            if path.suffix.lower() in {".csv", ".txt"}:
                import_opts = options or DataImportOptions(sample_rate=fs)
                signal = load_signal_file(path, options=import_opts)
            else:
                signal = load_signal_file(path, sample_rate=fs)
            signals.append(signal)
        return _combine_signals_as_channels(paths, signals)

    def _ask_import_options(self, path: Path) -> DataImportOptions | None:
        fs = frequency_to_hz(
            self.right_panel.sample_rate_input.value(),
            self.right_panel.sample_rate_unit.currentText(),
        )
        dialog = ImportDialog(
            path, fs, self, ai_config=self.right_panel.ai_config(),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.selected_options()

    def _accept_loaded_signal(self, signal: MultiChannelSignal, first_path: Path, file_count: int) -> None:
        self.signal = signal
        self.current_path = first_path
        self.processed_signal = None
        self.feature_rows = []
        self.last_auto_adjustment = ""
        self._narrowband_result = None

        self.right_panel.sample_rate_input.setValue(
            hz_to_frequency(self.signal.sample_rate, self.right_panel.sample_rate_unit.currentText())
        )
        self._auto_tune_filter_defaults()
        self._auto_tune_wavelet_defaults()
        self.left_panel.update_signal(
            first_path if file_count == 1 else f"{file_count} files",
            self.signal,
        )
        self.display.feature_widget.clear()
        self._refresh_time_plot()
        self._set_step(1)
        self.right_panel.set_ready("导入完成：已显示时域信号")
        self._update_status(
            state="导入完成",
            file=Path(first_path).name,
            rate=format_rate(self.signal.sample_rate),
            channels=f"{len(self.signal.channels)}",
            view="时域信号",
        )

    def _clear_current(self) -> None:
        self._cancel_all_workers()
        self.signal = None
        self.processed_signal = None
        self.current_path = None
        self.feature_rows = []
        self.last_auto_adjustment = ""
        self._narrowband_result = None
        self.left_panel.clear()
        self.display.clear_all()
        self._set_step(0)
        self.right_panel.set_ready("已清除当前内容")
        self._update_status(state="就绪", file="—", rate="—", channels="—", view="—")

    def _cancel_all_workers(self) -> None:
        for w in (self._cwt_worker, self._filter_worker, self._feature_worker):
            if w is not None and w.isRunning():
                w.cancel()
                w.wait(2000)

    # ── Plot refresh ──

    def _refresh_time_plot(self) -> None:
        if self.signal is None:
            self.display.show_empty()
            return
        visible = self._visible_data()
        if not visible:
            self._update_status(state="没有勾选需要显示的通道")
            return

        plot_channels = self._channels_for_time_display(visible)
        colors = self.left_panel.channel_colors()

        if self.right_panel.toneburst_preview_check.isChecked():
            amplitude = max(float(np.max(np.abs(values))) for values in visible.values())
            toneburst = generate_toneburst_preview(
                self.signal.time,
                self.right_panel.center_freq_hz(),
                self.right_panel.filter_order_input.value(),
                amplitude=amplitude if amplitude > 0 else 1.0,
            )
            plot_channels["Toneburst"] = toneburst
            colors["Toneburst"] = "#10B981"

        time, unit = self._display_time()
        self.display.plot_time_multi(
            time, plot_channels, colors,
            normalize=self.right_panel.normalize_check.isChecked(),
            stacked=self.right_panel.stacked_check.isChecked(),
            grid=self.right_panel.grid_check.isChecked(),
            time_unit=unit,
        )
        self._update_status(view="时域信号")

    def _channels_for_time_display(self, visible: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.processed_signal is not None and self.right_panel.filtered_envelope_check.isChecked():
            return {
                f"{name} 包络": _smooth_envelope(values, self.signal.sample_rate)
                for name, values in visible.items()
            }
        return dict(visible)

    # ── Narrowband wave packet extraction ──

    def _run_narrowband_extraction(self) -> None:
        if not self._require_signal():
            return
        try:
            center_hz = self.right_panel.center_freq_hz()
            bw_hz = self.right_panel.bandwidth_hz()
            window_s = self.right_panel.window_length_s()
            center_s = self.right_panel.manual_center_time_s() if not self.right_panel.auto_locate_check.isChecked() else None

            # Validate
            from core.filtering import validate_bandpass_params
            lowcut, highcut = validate_bandpass_params(
                self.signal.sample_rate, center_hz, bw_hz,
            )
            self.right_panel.lowcut_input.setValue(
                hz_to_frequency(lowcut, self.right_panel.lowcut_unit.currentText())
            )
            self.right_panel.highcut_input.setValue(
                hz_to_frequency(highcut, self.right_panel.highcut_unit.currentText())
            )
        except ValueError as exc:
            self._show_error("参数错误", str(exc))
            return

        visible = self._visible_data()
        if not visible:
            self._show_error("缺少数据", "请先勾选至少一个通道。")
            return

        name, values = next(iter(visible.items()))

        self.right_panel.set_busy(True, "正在进行窄带波包提取...")
        self._update_status(state="正在进行窄带波包提取...")

        self._filter_worker = FilteringWorker(
            self.signal.time, values, self.signal.sample_rate,
            center_freq=center_hz,
            bandwidth=bw_hz,
            order=self.right_panel.filter_order_input.value(),
            zero_phase=self.right_panel.zero_phase_check.isChecked(),
            remove_dc=True,
            auto_locate=self.right_panel.auto_locate_check.isChecked(),
            center_time=center_s,
            window_length=window_s,
            window_type=self.right_panel.window_type_combo.currentText(),
            output_mode=self.right_panel.output_mode_combo.currentText(),
            normalization=self.right_panel.normalization_combo.currentText(),
        )
        self._filter_worker.finished.connect(self._on_narrowband_done)
        self._filter_worker.error.connect(self._on_narrowband_error)
        self._filter_worker.start()

    def _on_narrowband_done(self, result: dict) -> None:
        self._narrowband_result = result
        self._set_step(2)

        time_unit = self.right_panel.time_unit_combo.currentText()
        freq_unit = self.right_panel.freq_display_unit.currentText()
        peak_t = format_time(result["peak_time"], time_unit)
        lowcut_str = format_frequency(result["lowcut"], freq_unit)
        highcut_str = format_frequency(result["highcut"], freq_unit)

        # Slice envelope to match packet time range
        envelope = result["envelope_full"] if self.right_panel.show_envelope_check.isChecked() else None
        if envelope is not None and envelope.size != result["time"].size:
            mask = (self.signal.time >= result["time"][0]) & (self.signal.time <= result["time"][-1])
            envelope = envelope[mask]

        self.display.plot_narrowband_result(
            result["time"], result["signal"],
            envelope=envelope,
            original_time=self.signal.time,
            original_signal=next(iter(self._visible_data().values())) if self.right_panel.show_original_check.isChecked() else None,
            filtered_full=result["filtered_full"] if self.right_panel.show_filtered_full_check.isChecked() else None,
            show_envelope=self.right_panel.show_envelope_check.isChecked(),
            show_original=self.right_panel.show_original_check.isChecked(),
            show_filtered_full=self.right_panel.show_filtered_full_check.isChecked(),
            time_unit=time_unit,
            grid=self.right_panel.grid_check.isChecked(),
        )

        self.right_panel.set_ready("窄带波包提取完成")
        self._update_status(
            state="窄带波包提取完成",
            view="时域信号",
        )
        self.statusBar().showMessage(
            f"窄带波包提取完成 | 峰值时间 = {peak_t} | 频带 = {lowcut_str}–{highcut_str}"
        )

    def _on_narrowband_error(self, msg: str) -> None:
        self.right_panel.set_ready("提取失败")
        self._show_error("窄带波包提取失败", msg)

    # ── Spectrum ──

    def _run_fft(self) -> None:
        if not self._require_signal():
            return
        try:
            spectra = {}
            for name, values in self._visible_data().items():
                freqs, amplitudes = compute_fft(values, self.signal.sample_rate)
                dom = find_dominant_frequency(values, self.signal.sample_rate, exclude_dc=True)
                spectra[name] = (freqs, amplitudes, float(dom["dominant_hz"]))
            self.display.plot_spectrum_multi(
                spectra, self.left_panel.channel_colors(),
                grid=self.right_panel.grid_check.isChecked(),
                ignore_dc=self.right_panel.ignore_dc_check.isChecked(),
                db_scale=self.right_panel.db_scale_check.isChecked(),
                freq_unit=self.right_panel.freq_display_unit.currentText(),
            )
        except Exception as exc:
            self._show_error("频谱分析失败", _friendly_error(exc))
            return
        self._set_step(2)
        self._update_status(state="频谱已生成", view="频域信号")

    # ── Features ──

    def _run_features(self) -> None:
        if not self._require_signal():
            return
        visible = self._visible_data()
        if not visible:
            self._show_error("缺少数据", "请先勾选至少一个通道。")
            return

        self.right_panel.set_busy(True, "正在计算特征参数...")
        self._feature_worker = FeatureWorker(self.signal.time, visible, self.signal.sample_rate)
        self._feature_worker.finished.connect(self._on_features_done)
        self._feature_worker.error.connect(self._on_features_error)
        self._feature_worker.start()

    def _on_features_done(self, rows: list[dict]) -> None:
        self.feature_rows = rows
        self.display.feature_widget.set_features(rows)
        self.display.show_features()
        self._set_step(3)
        self.right_panel.set_ready("特征计算完成")
        self._update_status(state="特征计算完成", view="特征参数")

    def _on_features_error(self, msg: str) -> None:
        self.right_panel.set_ready("特征计算失败")
        self._show_error("特征计算失败", msg)

    # ── Wavelet ──

    def _run_wavelet(self) -> None:
        if not self._require_signal():
            return
        visible = self._visible_data()
        if not visible:
            self._show_error("小波变换失败", "请先勾选至少一个通道。")
            return

        name, values = next(iter(visible.items()))
        f_min = self.right_panel.cwt_f_min_hz()
        f_max = self.right_panel.cwt_f_max_hz()
        n_freqs = self.right_panel.cwt_points_input.value()
        max_pts = self.right_panel.cwt_max_points_input.value()

        # Prepare signal for CWT
        time_range = None
        if self.right_panel.cwt_time_mode.currentText() == "自动定位主波包":
            if self._narrowband_result is not None:
                t0 = self._narrowband_result["peak_time"]
                half = self._narrowband_result["params"]["window_length"] * 2.0
                time_range = (t0 - half, t0 + half)

        prep = prepare_signal_for_cwt(
            self.signal.time, values, self.signal.sample_rate,
            time_range=time_range,
            max_points=max_pts if self.right_panel.cwt_auto_decimate.isChecked() else 999999999,
        )
        cost = estimate_cwt_cost(prep["signal"].size, n_freqs)
        self.right_panel.update_cwt_cost_info(
            original=values.size,
            input_pts=prep["signal"].size,
            decimation=prep["decimation_factor"],
            cost=cost,
        )

        self.right_panel.set_busy(True, "正在进行小波变换...")
        self._update_status(state="正在进行小波变换...")

        self._cwt_worker = CWTWorker(
            prep["signal"], prep["fs"], f_min, f_max, n_freqs,
            wavelet=self.right_panel.wavelet_combo.currentText(),
        )
        self._cwt_worker.progress.connect(
            lambda p: self._update_status(state=f"小波变换 {p}%")
        )
        self._cwt_worker.finished.connect(
            lambda r: self._on_wavelet_done(r, prep["time"])
        )
        self._cwt_worker.error.connect(self._on_wavelet_error)
        self._cwt_worker.start()

    def _cancel_wavelet(self) -> None:
        if self._cwt_worker is not None and self._cwt_worker.isRunning():
            self._cwt_worker.cancel()
            self.right_panel.set_ready("小波计算已取消")
            self._update_status(state="小波计算已取消")

    def _on_wavelet_done(self, result: dict, cwt_time: np.ndarray) -> None:
        freq_unit = self.right_panel.freq_display_unit.currentText()
        time_unit = self.right_panel.time_unit_combo.currentText()
        self.display.plot_wavelet(
            result["coefficients"], cwt_time, result["frequencies"],
            freq_unit=freq_unit,
            time_unit=time_unit,
            colormap=self.right_panel.cwt_colormap.currentText(),
        )
        self._set_step(3)
        self.right_panel.set_ready("小波图已生成")
        self._update_status(state="小波变换完成", view="小波变换")

    def _on_wavelet_error(self, msg: str) -> None:
        self.right_panel.set_ready("小波变换失败")
        self._update_status(state="小波变换失败")
        if "已取消" not in msg:
            self._show_error("小波变换失败", msg)

    # ── AI ──

    def _run_ai_adjustment(self) -> None:
        if not self._require_signal():
            return
        config = self.right_panel.ai_config()
        try:
            if config.enabled:
                if not config.api_key:
                    raise ValueError("请填写 API Key，或关闭「启用大模型 API」使用本地自动建议。")
                try:
                    suggestion = suggest_analysis_parameters(
                        self.signal.time, self.signal.channels, self.signal.sample_rate, config,
                    )
                except Exception as exc:
                    suggestion = self._local_parameter_suggestion()
                    suggestion["reason"] = f"在线 API 无法连接，已使用本地自动建议。{_friendly_error(exc)}"
            else:
                suggestion = self._local_parameter_suggestion()
            message = self._apply_parameter_suggestion(suggestion)
        except Exception as exc:
            self._show_error("智能识别参数失败", _friendly_error(exc))
            return
        QMessageBox.information(self, "智能识别参数", message)
        self.right_panel.set_ready(message)
        self._update_status(state="参数已调整")

    # ── Export ──

    def _export_narrowband_csv(self) -> None:
        result = self._narrowband_result
        if result is None:
            self._show_error("导出失败", "请先执行窄带波包提取。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出窄带波包 CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            export_signal_csv(
                MultiChannelSignal(
                    name="narrowband",
                    time=result["time"],
                    channels={"wave_packet": result["signal"]},
                    sample_rate=self.signal.sample_rate,
                ),
                path,
                ["wave_packet"],
            )
            self._set_step(4)
            self._update_status(state="导出完成", view="结果导出")
        except Exception as exc:
            self._show_error("导出失败", _friendly_error(exc))

    def _export_filtered_csv(self) -> None:
        result = self._narrowband_result
        if result is None:
            self._show_error("导出失败", "请先执行窄带波包提取。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出滤波后信号 CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            export_signal_csv(
                MultiChannelSignal(
                    name="filtered_full",
                    time=self.signal.time,
                    channels={"filtered": result["filtered_full"]},
                    sample_rate=self.signal.sample_rate,
                ),
                path,
                ["filtered"],
            )
            self._update_status(state="导出完成", view="结果导出")
        except Exception as exc:
            self._show_error("导出失败", _friendly_error(exc))

    def _export_envelope_csv(self) -> None:
        result = self._narrowband_result
        if result is None:
            self._show_error("导出失败", "请先执行窄带波包提取。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 Hilbert 包络 CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            export_signal_csv(
                MultiChannelSignal(
                    name="envelope",
                    time=self.signal.time,
                    channels={"envelope": result["envelope_full"]},
                    sample_rate=self.signal.sample_rate,
                ),
                path,
                ["envelope"],
            )
            self._update_status(state="导出完成", view="结果导出")
        except Exception as exc:
            self._show_error("导出失败", _friendly_error(exc))

    def _export_features_csv(self) -> None:
        if not self.display.feature_widget.has_data():
            self._show_error("导出失败", "请先计算特征参数。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出特征表 CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            self.display.feature_widget.export_csv(path)
        except Exception as exc:
            self._show_error("导出失败", _friendly_error(exc))
            return
        self._set_step(4)
        self._update_status(state="导出完成", view="结果导出")

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
        self._update_status(state="导出完成", view="结果导出")

    # ── Helpers ──

    def _visible_data(self) -> dict[str, np.ndarray]:
        source = self.processed_signal or self.signal
        if source is None:
            return {}
        visible = self.left_panel.visible_channels()
        return {name: source.channels[name] for name in visible if name in source.channels}

    def _display_time(self) -> tuple[np.ndarray, str]:
        assert self.signal is not None
        unit = self.right_panel.time_unit_combo.currentText()
        scale = {"s": 1.0, "ms": 1e3, "us": 1e6}[unit]
        return self.signal.time * scale, unit

    def _require_signal(self) -> bool:
        if self.signal is not None:
            return True
        self._show_error("缺少数据", "请先导入 CSV/TXT 或其他信号文件。")
        return False

    def _set_step(self, active_index: int) -> None:
        for i, label in enumerate(self.step_labels):
            label.setProperty("active", i <= active_index)
            label.style().unpolish(label)
            label.style().polish(label)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)
        self._update_status(state="错误")

    # ── Validation ──

    def _validate_filter(self) -> None:
        fs = self.signal.sample_rate
        lowcut, highcut = self.right_panel.center_band()
        nyquist = fs / 2.0
        self.last_auto_adjustment = ""
        if lowcut <= 0 or highcut >= nyquist:
            lowcut, highcut = self._auto_adjust_filter_band(fs)
            self.last_auto_adjustment = f"已自动调整滤波范围到 {lowcut:.6g} Hz – {highcut:.6g} Hz"
        self.right_panel.lowcut_input.setValue(
            hz_to_frequency(lowcut, self.right_panel.lowcut_unit.currentText())
        )
        self.right_panel.highcut_input.setValue(
            hz_to_frequency(highcut, self.right_panel.highcut_unit.currentText())
        )

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
        bw_hz = self.right_panel.bandwidth_hz()
        bandwidth = min(bw_hz, max_bandwidth)
        bandwidth = max(bandwidth, min(max_bandwidth, nyquist * 0.1))

        min_center = bandwidth / 2.0 + margin
        max_center = nyquist - bandwidth / 2.0 - margin
        if max_center < min_center:
            bandwidth = max(nyquist * 0.5, margin)
            min_center = bandwidth / 2.0 + margin
            max_center = nyquist - bandwidth / 2.0 - margin

        center = min(max(self.right_panel.center_freq_hz(), min_center), max_center)
        self.right_panel.set_center_freq_hz(center)
        self.right_panel.set_bandwidth_hz(bandwidth)
        return center - bandwidth / 2.0, center + bandwidth / 2.0

    def _validate_wavelet(self) -> None:
        fs = self.signal.sample_rate
        f_min = self.right_panel.cwt_f_min_hz()
        f_max = self.right_panel.cwt_f_max_hz()
        nyquist = fs / 2.0
        self.last_auto_adjustment = ""
        if f_min <= 0 or f_max >= nyquist or f_min >= f_max:
            f_min, f_max = self._auto_adjust_wavelet_range(fs)
            self.last_auto_adjustment = f"已自动调整小波频率范围到 {f_min:.6g} Hz – {f_max:.6g} Hz"

    def _auto_adjust_wavelet_range(self, fs: float) -> tuple[float, float]:
        nyquist = fs / 2.0
        max_freq = max(nyquist * 0.95, 1e-9)
        min_freq = max(min(self.right_panel.cwt_f_min_hz(), max_freq * 0.5), 1e-9)
        current_max = self.right_panel.cwt_f_max_hz()
        if current_max <= min_freq or current_max >= nyquist:
            current_max = max_freq
        if min_freq >= current_max:
            min_freq = max(current_max * 0.05, 1e-9)
        self.right_panel.cwt_min_input.setValue(
            hz_to_frequency(min_freq, self.right_panel.cwt_min_unit.currentText())
        )
        self.right_panel.cwt_max_input.setValue(
            hz_to_frequency(current_max, self.right_panel.cwt_max_unit.currentText())
        )
        return min_freq, current_max

    # ── Local AI suggestions ──

    def _local_parameter_suggestion(self) -> dict[str, float | int | str]:
        assert self.signal is not None
        doms = []
        for values in self.signal.channels.values():
            d = find_dominant_frequency(values, self.signal.sample_rate, exclude_dc=True)
            if d["dominant_hz"] > 0:
                doms.append(d["dominant_hz"])
        nyquist = self.signal.sample_rate / 2.0
        center = float(np.median(doms)) if doms else nyquist * 0.2
        center = min(max(center, nyquist * 0.02), nyquist * 0.8)
        bandwidth = min(max(center * 0.5, nyquist * 0.05), nyquist * 0.5)
        return {
            "center_frequency_hz": center,
            "bandwidth_hz": bandwidth,
            "filter_cycles": 3,
            "wavelet": "cmor1.5-1.0",
            "cwt_min_hz": max(center - bandwidth, nyquist * 0.01),
            "cwt_max_hz": min(center + bandwidth * 2.0, nyquist * 0.95),
            "cwt_points": self.right_panel.cwt_points_input.value(),
            "reason": "根据当前信号主频自动估计。",
        }

    def _apply_parameter_suggestion(self, suggestion: dict) -> str:
        assert self.signal is not None
        nyquist = self.signal.sample_rate / 2.0
        center = _coerce_float(suggestion.get("center_frequency_hz"), self.right_panel.center_freq_hz())
        bandwidth = _coerce_float(suggestion.get("bandwidth_hz"), self.right_panel.bandwidth_hz())
        center = min(max(center, nyquist * 0.02), nyquist * 0.9)
        bandwidth = min(max(bandwidth, nyquist * 0.01), nyquist * 0.8)
        self.right_panel.set_center_freq_hz(center)
        self.right_panel.set_bandwidth_hz(bandwidth)

        wavelet = str(suggestion.get("wavelet") or self.right_panel.wavelet_combo.currentText())
        idx = self.right_panel.wavelet_combo.findText(wavelet)
        if idx >= 0:
            self.right_panel.wavelet_combo.setCurrentIndex(idx)

        freq_unit = self.right_panel.freq_display_unit.currentText()
        cwt_min = _coerce_float(suggestion.get("cwt_min_hz"), self.right_panel.cwt_f_min_hz())
        cwt_max = _coerce_float(suggestion.get("cwt_max_hz"), self.right_panel.cwt_f_max_hz())
        self.right_panel.cwt_min_input.setValue(hz_to_frequency(max(cwt_min, 1e-9), self.right_panel.cwt_min_unit.currentText()))
        self.right_panel.cwt_max_input.setValue(hz_to_frequency(max(cwt_max, 1e-9), self.right_panel.cwt_max_unit.currentText()))

        reason = str(suggestion.get("reason") or "已根据当前信号自动调整。")
        return f"参数已调整：中心频率 {format_frequency(center, freq_unit)}，带宽 {format_frequency(bandwidth, freq_unit)}。{reason}"


# ── Module-level helpers ──

def _combine_signals_as_channels(paths: list[Path], signals: list[MultiChannelSignal]) -> MultiChannelSignal:
    if not signals:
        raise ValueError("No signals were loaded.")
    base_fs = signals[0].sample_rate
    for sig in signals[1:]:
        if abs(sig.sample_rate - base_fs) / base_fs > 1e-3:
            raise ValueError("多文件采样率不一致，请分别导入或统一采样率后再合并。")
    min_len = min(sig.time.size for sig in signals)
    channels: dict[str, np.ndarray] = {}
    for path, sig in zip(paths, signals):
        for ch_name, vals in sig.channels.items():
            if len(channels) >= 8:
                break
            merged = f"{path.stem}:{ch_name}" if len(signals) > 1 else str(ch_name)
            channels[merged] = vals[:min_len]
        if len(channels) >= 8:
            break
    return MultiChannelSignal(
        name=f"{len(paths)} files",
        time=signals[0].time[:min_len],
        channels=channels,
        sample_rate=base_fs,
        metadata={"source_files": [str(p) for p in paths], "truncated_samples": min_len},
    )


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _smooth_envelope(values: np.ndarray, sample_rate: float) -> np.ndarray:
    env = hilbert_envelope(values)
    win = max(int(sample_rate * 2e-6), 5)
    win = min(win, max(env.size // 20, 5))
    if win % 2 == 0:
        win += 1
    if win >= env.size:
        return env
    kernel = np.hanning(win)
    kernel /= np.sum(kernel)
    return np.convolve(env, kernel, mode="same")


def _friendly_error(error: Exception | str) -> str:
    if isinstance(error, _UserCancelled):
        return "已取消导入。"
    text = str(error)
    if "sample_rate is required" in text:
        return "无法识别采样率，请在导入窗口或右侧采样率输入框中手动填写采样率。"
    if "Nyquist" in text or "highcut" in text:
        return "滤波上限频率不能超过 Nyquist 频率，软件会优先尝试自动调整。"
    if "Unsupported file type" in text:
        return "当前文件格式暂不支持。建议使用 CSV/TXT、Excel、NPY/NPZ 或 MAT 文件。"
    return text


class _UserCancelled(Exception):
    """Internal marker for cancelled import dialogs."""
