"""Right-side parameter and action panel with scroll support."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.ui_helpers import make_primary_button, make_secondary_button
from core.ai_assistant import AIConfig, load_local_ai_config


class RightPanel(QWidget):
    """Scrollable parameter panel with card layout."""

    filter_requested = Signal()
    fft_requested = Signal()
    features_requested = Signal()
    wavelet_requested = Signal()
    wavelet_cancel_requested = Signal()
    export_signal_requested = Signal()
    export_filtered_signal_requested = Signal()
    export_envelope_requested = Signal()
    export_features_requested = Signal()
    export_image_requested = Signal()
    display_changed = Signal()
    ai_adjust_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidePanel")
        self.setMinimumWidth(300)
        self.setMaximumWidth(340)
        local_ai = load_local_ai_config(enabled=False)

        # --- Basic params ---
        self.sample_rate_input = self._double_input(1_000_000.0)
        self.sample_rate_unit = QComboBox()
        self.sample_rate_unit.addItems(["Hz", "kHz", "MHz"])
        self.sample_rate_unit.setCurrentText("Hz")
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(["s", "ms", "us"])
        self.time_unit_combo.setCurrentText("us")
        self.amplitude_unit_combo = QComboBox()
        self.amplitude_unit_combo.addItems(["V", "mV"])

        # --- Shared display frequency unit ---
        self.freq_display_unit = QComboBox()
        self.freq_display_unit.addItems(["Hz", "kHz", "MHz"])
        self.freq_display_unit.setCurrentText("kHz")

        # --- Filter params (internal always Hz) ---
        self.center_input = self._double_input(500.0)
        self.center_unit = QComboBox()
        self.center_unit.addItems(["Hz", "kHz", "MHz"])
        self.center_unit.setCurrentText("kHz")
        self.bandwidth_input = self._double_input(100.0)
        self.bandwidth_unit = QComboBox()
        self.bandwidth_unit.addItems(["Hz", "kHz", "MHz"])
        self.bandwidth_unit.setCurrentText("kHz")
        self.filter_order_input = QSpinBox()
        self.filter_order_input.setRange(1, 12)
        self.filter_order_input.setValue(4)
        self.lowcut_input = self._double_input(450.0)
        self.lowcut_unit = QComboBox()
        self.lowcut_unit.addItems(["Hz", "kHz", "MHz"])
        self.lowcut_unit.setCurrentText("kHz")
        self.highcut_input = self._double_input(550.0)
        self.highcut_unit = QComboBox()
        self.highcut_unit.addItems(["Hz", "kHz", "MHz"])
        self.highcut_unit.setCurrentText("kHz")
        self.zero_phase_check = QCheckBox("零相位滤波")
        self.zero_phase_check.setChecked(True)
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["Butterworth"])

        # --- Narrowband wave packet params ---
        self.auto_locate_check = QCheckBox("自动定位主波包中心")
        self.auto_locate_check.setChecked(True)
        self.manual_center_time = self._double_input(0.0)
        self.manual_center_time.setEnabled(False)
        self.center_time_unit = QComboBox()
        self.center_time_unit.addItems(["s", "ms", "us"])
        self.center_time_unit.setCurrentText("us")
        self.center_time_unit.setEnabled(False)
        self.window_length_input = self._double_input(0.0)
        self.window_length_input.setSpecialValueText("自动")
        self.window_length_unit = QComboBox()
        self.window_length_unit.addItems(["s", "ms", "us"])
        self.window_length_unit.setCurrentText("us")
        self.window_type_combo = QComboBox()
        self.window_type_combo.addItems(["tukey", "hann", "hamming", "none"])
        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItems(["segment", "full_zero"])
        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems(["max_abs", "none", "rms"])
        self.show_envelope_check = QCheckBox("叠加显示 Hilbert 包络")
        self.show_envelope_check.setChecked(True)
        self.show_original_check = QCheckBox("叠加原始信号（灰色）")
        self.show_filtered_full_check = QCheckBox("显示全长滤波信号")

        self.filtered_envelope_check = QCheckBox("窄带提取后默认显示包络")
        self.filtered_envelope_check.setChecked(True)

        # --- Wavelet params ---
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(["cmor1.5-1.0", "morl", "mexh", "gaus1", "gaus4"])
        self.cwt_min_input = self._double_input(100.0)
        self.cwt_min_unit = QComboBox()
        self.cwt_min_unit.addItems(["Hz", "kHz", "MHz"])
        self.cwt_min_unit.setCurrentText("kHz")
        self.cwt_max_input = self._double_input(450.0)
        self.cwt_max_unit = QComboBox()
        self.cwt_max_unit.addItems(["Hz", "kHz", "MHz"])
        self.cwt_max_unit.setCurrentText("kHz")
        self.cwt_points_input = QSpinBox()
        self.cwt_points_input.setRange(20, 300)
        self.cwt_points_input.setValue(100)
        self.cwt_time_mode = QComboBox()
        self.cwt_time_mode.addItems(["自动定位主波包", "使用全长信号"])
        self.cwt_max_points_input = QSpinBox()
        self.cwt_max_points_input.setRange(1000, 500000)
        self.cwt_max_points_input.setValue(30000)
        self.cwt_auto_decimate = QCheckBox("自动降采样")
        self.cwt_auto_decimate.setChecked(True)
        self.cwt_colormap = QComboBox()
        self.cwt_colormap.addItems(["viridis", "turbo", "jet", "plasma", "gray"])
        self.cwt_cost_label = QLabel("")
        self.cwt_cost_label.setObjectName("infoLabel")
        self.cwt_cost_label.setWordWrap(True)

        # --- Display settings ---
        self.normalize_check = QCheckBox("归一化显示")
        self.stacked_check = QCheckBox("叠加显示")
        self.grid_check = QCheckBox("显示网格")
        self.grid_check.setChecked(True)
        self.auto_unit_check = QCheckBox("自动单位换算")
        self.auto_unit_check.setChecked(True)
        self.ignore_dc_check = QCheckBox("去直流 / 忽略 DC 分量")
        self.ignore_dc_check.setChecked(True)
        self.toneburst_preview_check = QCheckBox("生成 Toneburst 预览")
        self.db_scale_check = QCheckBox("dB 幅值谱")
        self.db_scale_check.setChecked(False)

        # --- AI assistant ---
        self.ai_enable_check = QCheckBox("启用大模型 API")
        self.ai_base_url_input = QLineEdit(local_ai.base_url)
        self.ai_model_input = QLineEdit(local_ai.model)
        self.ai_key_input = QLineEdit(local_ai.api_key)
        self.ai_key_input.setEchoMode(QLineEdit.EchoMode.Password)

        # --- Status ---
        self.status_label = QLabel("就绪：请导入信号文件")
        self.status_label.setObjectName("statusHint")

        self._build_layout()
        self._connect_signals()

    # ── Getters (internal Hz / s) ──

    def center_freq_hz(self) -> float:
        from core.units import frequency_to_hz
        return frequency_to_hz(self.center_input.value(), self.center_unit.currentText())

    def bandwidth_hz(self) -> float:
        from core.units import frequency_to_hz
        return frequency_to_hz(self.bandwidth_input.value(), self.bandwidth_unit.currentText())

    def center_band(self) -> tuple[float, float]:
        center = self.center_freq_hz()
        half = self.bandwidth_hz() / 2.0
        return center - half, center + half

    def manual_center_time_s(self) -> float:
        from core.units import time_to_seconds
        return time_to_seconds(self.manual_center_time.value(), self.center_time_unit.currentText())

    def window_length_s(self) -> float | None:
        val = self.window_length_input.value()
        if val <= 0:
            return None
        from core.units import time_to_seconds
        return time_to_seconds(val, self.window_length_unit.currentText())

    def cwt_f_min_hz(self) -> float:
        from core.units import frequency_to_hz
        return frequency_to_hz(self.cwt_min_input.value(), self.cwt_min_unit.currentText())

    def cwt_f_max_hz(self) -> float:
        from core.units import frequency_to_hz
        return frequency_to_hz(self.cwt_max_input.value(), self.cwt_max_unit.currentText())

    # ── Setters (Hz → display) ──

    def set_center_freq_hz(self, hz: float) -> None:
        from core.units import hz_to_frequency
        self.center_input.setValue(hz_to_frequency(hz, self.center_unit.currentText()))

    def set_bandwidth_hz(self, hz: float) -> None:
        from core.units import hz_to_frequency
        self.bandwidth_input.setValue(hz_to_frequency(hz, self.bandwidth_unit.currentText()))

    # ── Busy / ready ──

    def set_busy(self, busy: bool, message: str) -> None:
        self.status_label.setText(message)
        for btn in self.findChildren(QPushButton):
            btn.setEnabled(not busy)

    def set_ready(self, message: str) -> None:
        self.set_busy(False, message)

    def ai_config(self) -> AIConfig:
        return AIConfig(
            enabled=self.ai_enable_check.isChecked(),
            api_key=self.ai_key_input.text().strip(),
            base_url=self.ai_base_url_input.text().strip() or "https://api.deepseek.com/v1",
            model=self.ai_model_input.text().strip() or "deepseek-chat",
        )

    def update_cwt_cost_info(self, original: int, input_pts: int, decimation: int, cost: int) -> None:
        lines = [
            f"原始点数：{original:,}",
            f"小波输入点数：{input_pts:,}",
            f"频率点数：{self.cwt_points_input.value()}",
            f"预计计算量：{cost:,}",
        ]
        if decimation > 1:
            lines.append(f"降采样倍数：{decimation}")
        self.cwt_cost_label.setText(" | ".join(lines))

    # ── Layout ──

    def _build_layout(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        content_layout.addWidget(self._build_basic_card())
        content_layout.addWidget(self._build_narrowband_card())
        content_layout.addWidget(self._build_wavelet_card())
        content_layout.addWidget(self._build_fft_card())
        content_layout.addWidget(self._build_display_card())
        content_layout.addWidget(self._build_ai_card())
        content_layout.addWidget(self._build_export_card())
        content_layout.addWidget(self.status_label)
        content_layout.addStretch(1)

        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _unit_row(self, spin, combo):
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(spin, 1)
        row.addWidget(combo)
        return row

    def _build_basic_card(self) -> QGroupBox:
        box = QGroupBox("基础参数")
        form = QFormLayout(box)
        form.setContentsMargins(10, 16, 10, 10)
        form.setVerticalSpacing(8)
        sr_row = QHBoxLayout()
        sr_row.setSpacing(4)
        sr_row.addWidget(self.sample_rate_input, 1)
        sr_row.addWidget(self.sample_rate_unit)
        form.addRow("采样率", sr_row)
        form.addRow("时间单位", self.time_unit_combo)
        form.addRow("频率显示单位", self.freq_display_unit)
        form.addRow("幅值单位", self.amplitude_unit_combo)
        return box

    def _build_narrowband_card(self) -> QGroupBox:
        box = QGroupBox("窄带波包提取")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 16, 10, 10)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setVerticalSpacing(8)
        form.addRow("中心频率", self._unit_row(self.center_input, self.center_unit))
        form.addRow("带宽", self._unit_row(self.bandwidth_input, self.bandwidth_unit))
        form.addRow("滤波器阶数", self.filter_order_input)

        # Advanced toggle
        adv_toggle = QToolButton()
        adv_toggle.setText("高级设置 ▸")
        adv_toggle.setCheckable(True)
        adv_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        adv_widget = QWidget()
        adv_form = QFormLayout(adv_widget)
        adv_form.setContentsMargins(0, 4, 0, 0)
        adv_form.setVerticalSpacing(8)
        adv_form.addRow("下限频率", self._unit_row(self.lowcut_input, self.lowcut_unit))
        adv_form.addRow("上限频率", self._unit_row(self.highcut_input, self.highcut_unit))
        adv_form.addRow("滤波器类型", self.filter_type_combo)
        adv_form.addRow(self.zero_phase_check)
        adv_widget.hide()
        adv_toggle.toggled.connect(lambda checked: (
            adv_toggle.setText("高级设置 ▾" if checked else "高级设置 ▸"),
            adv_widget.setVisible(checked),
        ))
        layout.addLayout(form)
        layout.addWidget(adv_toggle)
        layout.addWidget(adv_widget)

        # Wave packet settings
        layout.addWidget(self.auto_locate_check)
        center_time_row = self._unit_row(self.manual_center_time, self.center_time_unit)
        form2 = QFormLayout()
        form2.setVerticalSpacing(8)
        form2.addRow("手动中心时间", center_time_row)
        layout.addLayout(form2)

        win_len_row = self._unit_row(self.window_length_input, self.window_length_unit)
        form3 = QFormLayout()
        form3.setVerticalSpacing(8)
        form3.addRow("窗口长度", win_len_row)
        form3.addRow("窗函数", self.window_type_combo)
        form3.addRow("输出模式", self.output_mode_combo)
        form3.addRow("归一化", self.normalization_combo)
        layout.addLayout(form3)

        layout.addWidget(self.show_envelope_check)
        layout.addWidget(self.show_original_check)
        layout.addWidget(self.show_filtered_full_check)

        btn = make_primary_button("提取窄带波包")
        btn.clicked.connect(self.filter_requested.emit)
        layout.addWidget(btn)
        return box

    def _build_fft_card(self) -> QGroupBox:
        box = QGroupBox("频域分析")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 16, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(self.ignore_dc_check)
        layout.addWidget(self.db_scale_check)
        btn_row = QHBoxLayout()
        btn_fft = make_secondary_button("计算频谱")
        btn_fft.clicked.connect(self.fft_requested.emit)
        btn_row.addWidget(btn_fft)
        layout.addLayout(btn_row)
        return box

    def _build_wavelet_card(self) -> QGroupBox:
        box = QGroupBox("小波变换")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 16, 10, 10)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setVerticalSpacing(8)
        form.addRow("小波类型", self.wavelet_combo)
        form.addRow("最小频率", self._unit_row(self.cwt_min_input, self.cwt_min_unit))
        form.addRow("最大频率", self._unit_row(self.cwt_max_input, self.cwt_max_unit))
        form.addRow("频率点数", self.cwt_points_input)
        form.addRow("时间窗模式", self.cwt_time_mode)
        form.addRow("最大小波点数", self.cwt_max_points_input)
        form.addRow("色图", self.cwt_colormap)
        layout.addLayout(form)
        layout.addWidget(self.cwt_auto_decimate)
        layout.addWidget(self.cwt_cost_label)

        btn_row = QHBoxLayout()
        btn_wavelet = make_primary_button("生成小波图")
        btn_cancel = make_secondary_button("取消计算")
        btn_wavelet.clicked.connect(self.wavelet_requested.emit)
        btn_cancel.clicked.connect(self.wavelet_cancel_requested.emit)
        btn_row.addWidget(btn_wavelet, 1)
        btn_row.addWidget(btn_cancel, 1)
        layout.addLayout(btn_row)
        return box

    def _build_display_card(self) -> QGroupBox:
        box = QGroupBox("显示设置")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 16, 10, 10)
        layout.setSpacing(4)
        layout.addWidget(self.normalize_check)
        layout.addWidget(self.stacked_check)
        layout.addWidget(self.grid_check)
        layout.addWidget(self.auto_unit_check)
        layout.addWidget(self.toneburst_preview_check)
        layout.addWidget(self.filtered_envelope_check)
        btn_row = QHBoxLayout()
        btn_features = make_secondary_button("计算特征")
        btn_features.clicked.connect(self.features_requested.emit)
        btn_row.addWidget(btn_features)
        layout.addLayout(btn_row)
        return box

    def _build_ai_card(self) -> QGroupBox:
        box = QGroupBox("AI 辅助")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 16, 10, 10)
        layout.setSpacing(6)

        self.ai_collapse_toggle = QToolButton()
        self.ai_collapse_toggle.setText("AI 辅助 ▸")
        self.ai_collapse_toggle.setCheckable(True)
        self.ai_collapse_toggle.setChecked(False)
        self.ai_collapse_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        self.ai_inner = QWidget()
        ai_form = QFormLayout(self.ai_inner)
        ai_form.setContentsMargins(0, 4, 0, 0)
        ai_form.setVerticalSpacing(8)
        ai_form.addRow(self.ai_enable_check)
        ai_form.addRow("API 地址", self.ai_base_url_input)
        ai_form.addRow("模型", self.ai_model_input)
        ai_form.addRow("API Key", self.ai_key_input)
        btn = make_secondary_button("智能识别参数")
        btn.clicked.connect(self.ai_adjust_requested.emit)
        ai_form.addRow(btn)
        self.ai_inner.hide()

        self.ai_collapse_toggle.toggled.connect(lambda checked: (
            self.ai_collapse_toggle.setText("AI 辅助 ▾" if checked else "AI 辅助 ▸"),
            self.ai_inner.setVisible(checked),
        ))

        layout.addWidget(self.ai_collapse_toggle)
        layout.addWidget(self.ai_inner)
        return box

    def _build_export_card(self) -> QGroupBox:
        box = QGroupBox("结果导出")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 16, 10, 10)
        layout.setSpacing(6)
        b1 = make_secondary_button("导出窄带波包 CSV")
        b2 = make_secondary_button("导出滤波后信号 CSV")
        b3 = make_secondary_button("导出 Hilbert 包络 CSV")
        b4 = make_secondary_button("导出特征表 CSV")
        b5 = make_secondary_button("导出当前图像")
        b1.clicked.connect(self.export_signal_requested.emit)
        b2.clicked.connect(self.export_filtered_signal_requested.emit)
        b3.clicked.connect(self.export_envelope_requested.emit)
        b4.clicked.connect(self.export_features_requested.emit)
        b5.clicked.connect(self.export_image_requested.emit)
        layout.addWidget(b1)
        layout.addWidget(b2)
        layout.addWidget(b3)
        layout.addWidget(b4)
        layout.addWidget(b5)
        return box

    def _connect_signals(self) -> None:
        checkables = (
            self.normalize_check, self.stacked_check, self.grid_check,
            self.auto_unit_check, self.ignore_dc_check, self.toneburst_preview_check,
            self.filtered_envelope_check, self.db_scale_check, self.show_envelope_check,
            self.show_original_check, self.show_filtered_full_check,
        )
        for w in checkables:
            w.stateChanged.connect(lambda _state: self.display_changed.emit())
        self.time_unit_combo.currentTextChanged.connect(lambda _text: self.display_changed.emit())
        self.freq_display_unit.currentTextChanged.connect(lambda _text: self.display_changed.emit())
        self.auto_locate_check.toggled.connect(self._on_auto_locate_toggled)

    def _on_auto_locate_toggled(self, checked: bool) -> None:
        self.manual_center_time.setEnabled(not checked)
        self.center_time_unit.setEnabled(not checked)

    def _double_input(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0001, 10_000_000_000.0)
        spin.setDecimals(6)
        spin.setValue(value)
        return spin
