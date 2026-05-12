"""Right-side parameter and action panel."""

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
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.ui_helpers import make_primary_button, make_secondary_button
from core.ai_assistant import AIConfig


class ControlPanel(QWidget):
    """User-facing operation panel with simple defaults and advanced settings."""

    filter_requested = Signal()
    fft_requested = Signal()
    features_requested = Signal()
    wavelet_requested = Signal()
    export_signal_requested = Signal()
    export_features_requested = Signal()
    export_image_requested = Signal()
    display_changed = Signal()
    ai_adjust_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidePanel")

        self.sample_rate_input = self._double_input(1_000_000.0, " Hz")
        self.center_input = self._double_input(500_000.0, " Hz")
        self.bandwidth_input = self._double_input(100_000.0, " Hz")
        self.lowcut_input = self._double_input(450_000.0, " Hz")
        self.highcut_input = self._double_input(550_000.0, " Hz")
        self.filter_order_input = QSpinBox()
        self.filter_order_input.setRange(1, 12)
        self.filter_order_input.setValue(4)
        self.filter_cycles_input = QSpinBox()
        self.filter_cycles_input.setRange(1, 20)
        self.filter_cycles_input.setValue(3)
        self.zero_phase_check = QCheckBox("零相位滤波")
        self.zero_phase_check.setChecked(True)
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["Butterworth"])

        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(["morl", "cmor1.5-1.0", "mexh"])
        self.cwt_min_input = self._double_input(100_000.0, " Hz")
        self.cwt_max_input = self._double_input(900_000.0, " Hz")
        self.cwt_points_input = QSpinBox()
        self.cwt_points_input.setRange(8, 512)
        self.cwt_points_input.setValue(96)

        self.normalize_check = QCheckBox("归一化")
        self.stacked_check = QCheckBox("分通道错位显示")
        self.grid_check = QCheckBox("显示网格")
        self.grid_check.setChecked(True)
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(["s", "ms", "us"])

        self.ai_enable_check = QCheckBox("启用大模型 API")
        self.ai_base_url_input = QLineEdit("https://api.openai.com/v1")
        self.ai_model_input = QLineEdit("gpt-4o-mini")
        self.ai_key_input = QLineEdit()
        self.ai_key_input.setEchoMode(QLineEdit.EchoMode.Password)

        self.status_label = QLabel("等待导入数据")
        self.status_label.setObjectName("statusHint")

        self._build_layout()
        self._connect_signals()

    def set_busy(self, busy: bool, message: str) -> None:
        """Show operation status and disable repeated action buttons."""
        self.status_label.setText(message)
        for button in self.findChildren(QPushButton):
            button.setEnabled(not busy)

    def set_ready(self, message: str) -> None:
        """Show a non-busy status message."""
        self.set_busy(False, message)

    def center_band(self) -> tuple[float, float]:
        """Return low/high cutoffs from the simple center/bandwidth controls."""
        center = self.center_input.value()
        half = self.bandwidth_input.value() / 2.0
        return center - half, center + half

    def ai_config(self) -> AIConfig:
        """Return current optional AI API configuration."""
        return AIConfig(
            enabled=self.ai_enable_check.isChecked(),
            api_key=self.ai_key_input.text().strip(),
            base_url=self.ai_base_url_input.text().strip() or "https://api.openai.com/v1",
            model=self.ai_model_input.text().strip() or "gpt-4o-mini",
        )

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        processing_box = QGroupBox("处理参数")
        processing_form = QFormLayout(processing_box)
        processing_form.addRow("采样率", self.sample_rate_input)
        calculate_fft = make_secondary_button("查看频谱")
        calculate_fft.clicked.connect(self.fft_requested.emit)
        calculate_features = make_secondary_button("计算特征")
        calculate_features.clicked.connect(self.features_requested.emit)
        button_row = QHBoxLayout()
        button_row.addWidget(calculate_fft)
        button_row.addWidget(calculate_features)
        processing_form.addRow(button_row)
        layout.addWidget(processing_box)

        filter_box = QGroupBox("窄带提取与 Hilbert")
        filter_layout = QVBoxLayout(filter_box)
        filter_form = QFormLayout()
        filter_form.addRow("中心频率", self.center_input)
        filter_form.addRow("带宽", self.bandwidth_input)
        filter_form.addRow("滤波后周期", self.filter_cycles_input)
        filter_layout.addLayout(filter_form)
        advanced_button = QToolButton()
        advanced_button.setText("高级设置")
        advanced_button.setCheckable(True)
        advanced_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        advanced_widget = QWidget()
        advanced_form = QFormLayout(advanced_widget)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.addRow("下限频率", self.lowcut_input)
        advanced_form.addRow("上限频率", self.highcut_input)
        advanced_form.addRow("滤波器阶数", self.filter_order_input)
        advanced_form.addRow("滤波器类型", self.filter_type_combo)
        advanced_form.addRow(self.zero_phase_check)
        advanced_widget.hide()
        advanced_button.toggled.connect(advanced_widget.setVisible)
        filter_layout.addWidget(advanced_button)
        filter_layout.addWidget(advanced_widget)
        filter_button = make_primary_button("开始窄带提取")
        filter_button.clicked.connect(self.filter_requested.emit)
        filter_layout.addWidget(filter_button)
        layout.addWidget(filter_box)

        wavelet_box = QGroupBox("小波变换")
        wavelet_form = QFormLayout(wavelet_box)
        wavelet_form.addRow("小波类型", self.wavelet_combo)
        wavelet_form.addRow("最小频率", self.cwt_min_input)
        wavelet_form.addRow("最大频率", self.cwt_max_input)
        wavelet_form.addRow("频率点数", self.cwt_points_input)
        wavelet_button = make_primary_button("生成小波图")
        wavelet_button.clicked.connect(self.wavelet_requested.emit)
        wavelet_form.addRow(wavelet_button)
        layout.addWidget(wavelet_box)

        display_box = QGroupBox("显示设置")
        display_form = QFormLayout(display_box)
        display_form.addRow(self.normalize_check)
        display_form.addRow(self.stacked_check)
        display_form.addRow(self.grid_check)
        display_form.addRow("时间单位", self.time_unit_combo)
        layout.addWidget(display_box)

        ai_box = QGroupBox("AI 辅助")
        ai_form = QFormLayout(ai_box)
        ai_form.addRow(self.ai_enable_check)
        ai_form.addRow("API Base", self.ai_base_url_input)
        ai_form.addRow("Model", self.ai_model_input)
        ai_form.addRow("API Key", self.ai_key_input)
        ai_button = make_secondary_button("智能识别参数")
        ai_button.clicked.connect(self.ai_adjust_requested.emit)
        ai_form.addRow(ai_button)
        layout.addWidget(ai_box)

        export_box = QGroupBox("结果导出")
        export_layout = QVBoxLayout(export_box)
        export_signal = make_secondary_button("导出处理后 CSV")
        export_features = make_secondary_button("导出特征表 CSV")
        export_image = make_secondary_button("导出当前图像")
        export_signal.clicked.connect(self.export_signal_requested.emit)
        export_features.clicked.connect(self.export_features_requested.emit)
        export_image.clicked.connect(self.export_image_requested.emit)
        export_layout.addWidget(export_signal)
        export_layout.addWidget(export_features)
        export_layout.addWidget(export_image)
        layout.addWidget(export_box)

        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def _connect_signals(self) -> None:
        for widget in (
            self.normalize_check,
            self.stacked_check,
            self.grid_check,
            self.time_unit_combo,
        ):
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self.display_changed.emit)
            else:
                widget.stateChanged.connect(self.display_changed.emit)

    def _double_input(self, value: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.001, 10_000_000_000.0)
        spin.setDecimals(3)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

