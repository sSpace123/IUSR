"""Small UI helpers shared by the desktop application."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton


def make_step_label(text: str, active: bool = False) -> QLabel:
    """Create one label in the top workflow guide."""
    label = QLabel(text)
    label.setProperty("active", active)
    return label


def make_primary_button(text: str) -> QPushButton:
    """Create a primary action button."""
    button = QPushButton(text)
    button.setProperty("variant", "primary")
    return button


def make_secondary_button(text: str) -> QPushButton:
    """Create a secondary action button."""
    button = QPushButton(text)
    button.setProperty("variant", "secondary")
    return button


def format_rate(value: float) -> str:
    """Format a sampling rate for display."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g} MHz"
    if value >= 1_000:
        return f"{value / 1_000:.3g} kHz"
    return f"{value:.3g} Hz"


def format_duration(seconds: float) -> str:
    """Format a signal duration for display."""
    if seconds >= 1:
        return f"{seconds:.3g} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3g} ms"
    return f"{seconds * 1e6:.3g} us"

