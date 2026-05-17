"""Unified stylesheet for the ultrasonic signal analyzer."""

COLORS = {
    "primary": "#2563EB",
    "primary_hover": "#1D4ED8",
    "primary_light": "#DBEAFE",
    "accent": "#10B981",
    "accent_light": "#D1FAE5",
    "warning": "#F59E0B",
    "warning_light": "#FEF3C7",
    "error": "#EF4444",
    "error_light": "#FEE2E2",
    "bg": "#F7F9FC",
    "card": "#FFFFFF",
    "border": "#E5E7EB",
    "text": "#111827",
    "muted": "#6B7280",
    "placeholder": "#9CA3AF",
}


def stylesheet() -> str:
    return f"""
    * {{
        font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
        font-size: 13px;
        color: {COLORS["text"]};
    }}

    QMainWindow, QWidget#centerPanel {{
        background: {COLORS["bg"]};
    }}

    /* ── Top bar ── */
    QFrame#topBar {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 8px 12px;
    }}

    QLabel#appTitle {{
        font-size: 18px;
        font-weight: 700;
        color: {COLORS["text"]};
    }}

    QLabel#stepLabel {{
        font-size: 13px;
        font-weight: 600;
    }}

    QLabel#stepLabel[active="true"] {{
        color: {COLORS["primary"]};
    }}

    QLabel#stepLabel[active="false"] {{
        color: {COLORS["muted"]};
    }}

    QLabel#stepArrow {{
        color: {COLORS["muted"]};
        font-size: 14px;
    }}

    /* ── Side panels ── */
    QWidget#sidePanel {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
    }}

    /* ── Cards / GroupBox ── */
    QGroupBox {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        margin-top: 16px;
        padding: 16px 12px 12px 12px;
        font-weight: 700;
        font-size: 13px;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: {COLORS["text"]};
    }}

    /* ── Buttons ── */
    QPushButton {{
        border-radius: 6px;
        padding: 8px 14px;
        font-weight: 600;
        font-size: 13px;
        min-height: 34px;
    }}

    QPushButton[variant="primary"] {{
        background: {COLORS["primary"]};
        color: white;
        border: 1px solid {COLORS["primary"]};
    }}

    QPushButton[variant="primary"]:hover {{
        background: {COLORS["primary_hover"]};
    }}

    QPushButton[variant="primary"]:disabled {{
        background: #93C5FD;
        border-color: #93C5FD;
    }}

    QPushButton[variant="secondary"] {{
        background: {COLORS["card"]};
        color: {COLORS["primary"]};
        border: 1px solid #BFDBFE;
    }}

    QPushButton[variant="secondary"]:hover {{
        background: {COLORS["primary_light"]};
    }}

    QPushButton[variant="secondary"]:disabled {{
        color: {COLORS["muted"]};
        border-color: {COLORS["border"]};
    }}

    QPushButton[variant="accent"] {{
        background: {COLORS["accent"]};
        color: white;
        border: 1px solid {COLORS["accent"]};
    }}

    QPushButton[variant="accent"]:hover {{
        background: #059669;
    }}

    /* ── Inputs ── */
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
        background: {COLORS["card"]};
        border: 1px solid #D1D5DB;
        border-radius: 4px;
        padding: 5px 8px;
        min-height: 30px;
        font-size: 13px;
    }}

    QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
        border-color: {COLORS["primary"]};
    }}

    QComboBox::drop-down {{
        border: 0;
        padding-right: 6px;
    }}

    QToolButton {{
        border: 1px solid {COLORS["border"]};
        border-radius: 4px;
        padding: 4px 10px;
        font-weight: 600;
        color: {COLORS["muted"]};
    }}

    QToolButton:checked {{
        background: {COLORS["primary_light"]};
        color: {COLORS["primary"]};
        border-color: {COLORS["primary"]};
    }}

    /* ── Tabs ── */
    QTabWidget::pane {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
    }}

    QTabBar::tab {{
        background: #EEF2FF;
        color: #374151;
        padding: 10px 20px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        margin-right: 3px;
        font-weight: 600;
        min-height: 36px;
    }}

    QTabBar::tab:selected {{
        background: {COLORS["primary"]};
        color: white;
    }}

    QTabBar::tab:hover:!selected {{
        background: #DBEAFE;
    }}

    /* ── Table ── */
    QTableWidget {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        gridline-color: #F3F4F6;
    }}

    QHeaderView::section {{
        background: #F3F4F6;
        border: 0;
        padding: 8px;
        font-weight: 700;
        font-size: 12px;
    }}

    /* ── Scroll area ── */
    QScrollArea {{
        border: 0;
        background: transparent;
    }}

    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background: #CBD5E1;
        border-radius: 4px;
        min-height: 30px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: #94A3B8;
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    QScrollBar:horizontal {{
        background: transparent;
        height: 8px;
        margin: 0;
    }}

    QScrollBar::handle:horizontal {{
        background: #CBD5E1;
        border-radius: 4px;
        min-width: 30px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background: #94A3B8;
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}

    /* ── Checkbox ── */
    QCheckBox {{
        spacing: 8px;
        font-size: 13px;
    }}

    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid #D1D5DB;
    }}

    QCheckBox::indicator:checked {{
        background: {COLORS["primary"]};
        border-color: {COLORS["primary"]};
    }}

    /* ── Status ── */
    QLabel#statusHint {{
        color: #047857;
        background: {COLORS["accent_light"]};
        border: 1px solid #A7F3D0;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 12px;
    }}

    QStatusBar {{
        background: {COLORS["card"]};
        border-top: 1px solid {COLORS["border"]};
        padding: 2px 8px;
        font-size: 12px;
        min-height: 28px;
    }}

    QStatusBar QLabel {{
        font-size: 12px;
        color: {COLORS["muted"]};
    }}

    /* ── Empty state ── */
    QFrame#emptyStateFrame {{
        background: {COLORS["card"]};
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
    }}

    QLabel#emptyStateTitle {{
        font-size: 18px;
        font-weight: 700;
        color: {COLORS["text"]};
    }}

    QLabel#emptyStateHint {{
        font-size: 14px;
        color: {COLORS["muted"]};
    }}

    /* ── Channel row ── */
    QFrame#channelRow {{
        background: transparent;
        border: 0;
    }}

    QFrame#channelRow:hover {{
        background: #F3F4F6;
        border-radius: 4px;
    }}

    /* ── Info labels ── */
    QLabel#infoValue {{
        font-weight: 600;
        color: {COLORS["text"]};
    }}

    QLabel#infoLabel {{
        color: {COLORS["muted"]};
    }}

    /* ── Splitter ── */
    QSplitter::handle {{
        background: {COLORS["border"]};
        margin: 0 2px;
    }}

    QSplitter::handle:horizontal {{
        width: 3px;
    }}
    """
