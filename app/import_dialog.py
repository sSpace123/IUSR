"""Import preview dialog for user-controlled tabular signal loading."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.ai_assistant import AIConfig, suggest_import_options
from core.data_loader import DataImportOptions, FilePreview, preview_signal_file


class ImportDialog(QDialog):
    """Let users review detected file structure and choose import columns."""

    def __init__(
        self,
        path: str | Path,
        default_sample_rate: float,
        parent: QWidget | None = None,
        ai_config: AIConfig | None = None,
    ):
        super().__init__(parent)
        self.path = Path(path)
        self.ai_config = ai_config or AIConfig()
        self.preview: FilePreview | None = None
        self.setWindowTitle("Import Preview")
        self.resize(780, 560)

        self.summary_label = QLabel()
        self.skip_rows_input = QSpinBox()
        self.skip_rows_input.setRange(0, 10_000_000)
        self.has_header_check = QCheckBox("First visible row contains column names")

        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItem("Auto", None)
        self.delimiter_combo.addItem("Comma (,)", ",")
        self.delimiter_combo.addItem("Tab", "\t")
        self.delimiter_combo.addItem("Whitespace", r"\s+")
        self.delimiter_combo.addItem("Semicolon (;)", ";")

        self.sample_rate_input = QDoubleSpinBox()
        self.sample_rate_input.setRange(0.001, 10_000_000_000.0)
        self.sample_rate_input.setDecimals(3)
        self.sample_rate_input.setValue(default_sample_rate)
        self.sample_rate_input.setSuffix(" Hz")

        self.time_column_combo = QComboBox()
        self.column_table = QTableWidget(0, 2)
        self.column_table.setHorizontalHeaderLabels(["Use", "Column"])
        self.column_table.horizontalHeader().setStretchLastSection(True)

        self.preview_table = QTableWidget()
        self.reload_button = QPushButton("Reload Preview")
        self.ai_button = QPushButton("智能识别文件结构")
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._loaded_once = False

        self._build_layout()
        self._connect_signals()
        self._load_preview()

    def selected_options(self) -> DataImportOptions:
        """Return selected import options."""
        value_columns: list[str] = []
        for row in range(self.column_table.rowCount()):
            checkbox = self.column_table.cellWidget(row, 0)
            item = self.column_table.item(row, 1)
            if isinstance(checkbox, QCheckBox) and checkbox.isChecked() and item is not None:
                value_columns.append(item.text())

        return DataImportOptions(
            skip_rows=self.skip_rows_input.value(),
            delimiter=self.delimiter_combo.currentData(),
            has_header=self.has_header_check.isChecked(),
            time_column=self.time_column_combo.currentData(),
            value_columns=value_columns,
            sample_rate=self.sample_rate_input.value(),
        )

    def accept(self) -> None:
        if not self.selected_options().value_columns:
            QMessageBox.warning(self, "Import Preview", "Select at least one signal column.")
            return
        super().accept()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)

        summary_box = QGroupBox("File")
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.addWidget(self.summary_label)
        layout.addWidget(summary_box)

        settings_box = QGroupBox("Import Settings")
        settings = QGridLayout(settings_box)
        settings.addWidget(QLabel("Skip rows"), 0, 0)
        settings.addWidget(self.skip_rows_input, 0, 1)
        settings.addWidget(QLabel("Delimiter"), 0, 2)
        settings.addWidget(self.delimiter_combo, 0, 3)
        settings.addWidget(self.has_header_check, 1, 0, 1, 2)
        settings.addWidget(QLabel("Sample rate"), 1, 2)
        settings.addWidget(self.sample_rate_input, 1, 3)
        settings.addWidget(QLabel("Time column"), 2, 0)
        settings.addWidget(self.time_column_combo, 2, 1)
        settings.addWidget(self.reload_button, 2, 2)
        settings.addWidget(self.ai_button, 2, 3)
        layout.addWidget(settings_box)

        table_row = QHBoxLayout()
        column_box = QGroupBox("Columns")
        column_layout = QVBoxLayout(column_box)
        column_layout.addWidget(self.column_table)
        preview_box = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.addWidget(self.preview_table)
        table_row.addWidget(column_box, 1)
        table_row.addWidget(preview_box, 3)
        layout.addLayout(table_row, 1)

        layout.addWidget(self.buttons)

    def _connect_signals(self) -> None:
        self.reload_button.clicked.connect(self._load_preview)
        self.ai_button.clicked.connect(self._run_ai_import_suggestion)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.time_column_combo.currentIndexChanged.connect(self._refresh_column_checks)

    def _load_preview(self) -> None:
        try:
            skip_rows = None if not self._loaded_once else self.skip_rows_input.value()
            has_header = None if not self._loaded_once else self.has_header_check.isChecked()
            self.preview = preview_signal_file(
                self.path,
                skip_rows=skip_rows,
                delimiter=self.delimiter_combo.currentData(),
                has_header=has_header,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Preview failed", str(exc))
            return
        self._loaded_once = True

        self.skip_rows_input.blockSignals(True)
        self.skip_rows_input.setValue(self.preview.skip_rows)
        self.skip_rows_input.blockSignals(False)
        self.has_header_check.setChecked(self.preview.has_header)

        size_kb = self.preview.file_size / 1024.0
        self.summary_label.setText(
            f"{self.preview.path.name}    {size_kb:.1f} KB    "
            f"{self.preview.total_lines or 0} lines"
        )
        self._fill_preview_table()
        self._fill_column_controls()

    def _fill_preview_table(self) -> None:
        if self.preview is None:
            return
        self.preview_table.clear()
        self.preview_table.setColumnCount(len(self.preview.columns))
        self.preview_table.setHorizontalHeaderLabels(self.preview.columns)
        self.preview_table.setRowCount(len(self.preview.preview_rows))
        for row_index, row in enumerate(self.preview.preview_rows):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.preview_table.setItem(row_index, column_index, item)
        self.preview_table.resizeColumnsToContents()

    def _fill_column_controls(self) -> None:
        if self.preview is None:
            return
        self.time_column_combo.blockSignals(True)
        self.time_column_combo.clear()
        self.time_column_combo.addItem("None", None)
        for column in self.preview.columns:
            self.time_column_combo.addItem(column, column)
        time_index = self._guess_time_column_index(self.preview.columns)
        self.time_column_combo.setCurrentIndex(time_index)
        self.time_column_combo.blockSignals(False)

        self.column_table.setRowCount(len(self.preview.columns))
        for row, column in enumerate(self.preview.columns):
            checkbox = QCheckBox()
            checkbox.setChecked(row != time_index - 1)
            self.column_table.setCellWidget(row, 0, checkbox)
            item = QTableWidgetItem(column)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.column_table.setItem(row, 1, item)
        self._refresh_column_checks()

    def _refresh_column_checks(self) -> None:
        time_column = self.time_column_combo.currentData()
        for row in range(self.column_table.rowCount()):
            checkbox = self.column_table.cellWidget(row, 0)
            item = self.column_table.item(row, 1)
            if not isinstance(checkbox, QCheckBox) or item is None:
                continue
            is_time_column = item.text() == time_column
            checkbox.setEnabled(not is_time_column)
            if is_time_column:
                checkbox.setChecked(False)

    def _guess_time_column_index(self, columns: list[str]) -> int:
        hints = {"time", "t", "timestamp", "sec", "second", "seconds", "s"}
        for index, column in enumerate(columns, start=1):
            if column.strip().lower() in hints:
                return index
        return 0

    def _run_ai_import_suggestion(self) -> None:
        if self.preview is None:
            return
        if not self.ai_config.enabled:
            QMessageBox.information(self, "AI Assistant", "请先在主界面右侧启用大模型 API。")
            return
        if not self.ai_config.api_key:
            QMessageBox.warning(self, "AI Assistant", "请先填写 API Key。")
            return
        try:
            suggestion = suggest_import_options(self.preview, self.ai_config)
            self._apply_ai_import_suggestion(suggestion)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "AI Assistant",
                f"AI API 连接失败，已保留当前本地识别结果。\n\n{exc}",
            )

    def _apply_ai_import_suggestion(self, suggestion: dict) -> None:
        if "skip_rows" in suggestion:
            self.skip_rows_input.setValue(int(float(suggestion["skip_rows"])))
        if "has_header" in suggestion:
            self.has_header_check.setChecked(bool(suggestion["has_header"]))
        if "sample_rate_hz" in suggestion and suggestion["sample_rate_hz"]:
            self.sample_rate_input.setValue(float(suggestion["sample_rate_hz"]))
        self._load_preview()

        time_column = str(suggestion.get("time_column") or "")
        index = self.time_column_combo.findText(time_column)
        if index >= 0:
            self.time_column_combo.setCurrentIndex(index)

        selected = {str(column) for column in suggestion.get("value_columns", [])}
        for row in range(self.column_table.rowCount()):
            checkbox = self.column_table.cellWidget(row, 0)
            item = self.column_table.item(row, 1)
            if isinstance(checkbox, QCheckBox) and item is not None and selected:
                checkbox.setChecked(item.text() in selected)
        QMessageBox.information(
            self,
            "AI Assistant",
            str(suggestion.get("reason") or "已根据大模型建议调整导入参数。"),
        )
