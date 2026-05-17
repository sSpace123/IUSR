"""Data loading utilities for CSV, TXT, Excel, NumPy, and optional MAT files."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .signal_data import MultiChannelSignal, infer_sample_rate_from_time

TIME_COLUMN_HINTS = {"time", "t", "timestamp", "sec", "second", "seconds", "s"}
DELIMITER_LABELS = {
    "auto": None,
    "comma": ",",
    "tab": "\t",
    "space": r"\s+",
    "semicolon": ";",
}


@dataclass
class DataImportOptions:
    """User-controlled options for importing tabular signal files."""

    skip_rows: int | None = None
    delimiter: str | None = None
    has_header: bool | None = None
    time_column: str | int | None = None
    value_columns: list[str | int] | None = None
    sample_rate: float | None = None


@dataclass
class FilePreview:
    """Small preview of an input file for import configuration."""

    path: Path
    file_size: int
    total_lines: int | None
    delimiter: str | None
    skip_rows: int
    has_header: bool
    columns: list[str]
    preview_rows: list[list[str]]
    metadata: dict[str, str] = field(default_factory=dict)


def load_signal_file(
    path: str | Path,
    sample_rate: float | None = None,
    options: DataImportOptions | None = None,
) -> MultiChannelSignal:
    """Load a signal file into a MultiChannelSignal.

    CSV and TXT are the primary MVP formats. XLSX, NPY, NPZ, and MAT are
    supported when their backing dependencies and file layouts are simple.
    Files without a time column require an explicit sample_rate.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    suffix = file_path.suffix.lower()
    import_options = options or DataImportOptions(sample_rate=sample_rate)
    if suffix in {".csv", ".txt"}:
        frame = _read_delimited_table(file_path, import_options)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(file_path)
    elif suffix == ".npy":
        frame = _frame_from_array(np.load(file_path, allow_pickle=False))
    elif suffix == ".npz":
        frame = _frame_from_npz(file_path)
    elif suffix == ".mat":
        frame = _frame_from_mat(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    effective_sample_rate = import_options.sample_rate
    return _frame_to_signal(
        file_path.stem,
        frame,
        effective_sample_rate,
        time_column=import_options.time_column,
        value_columns=import_options.value_columns,
    )


def preview_signal_file(
    path: str | Path,
    skip_rows: int | None = None,
    delimiter: str | None = None,
    has_header: bool | None = None,
    max_preview_rows: int = 20,
) -> FilePreview:
    """Inspect a tabular signal file and return import defaults plus preview rows."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if file_path.suffix.lower() not in {".csv", ".txt"}:
        raise ValueError("Preview is currently available for CSV and TXT files.")

    detected = _detect_table_layout(file_path, delimiter)
    selected_skip_rows = detected["skip_rows"] if skip_rows is None else max(skip_rows, 0)
    selected_delimiter = delimiter if delimiter is not None else detected["delimiter"]
    selected_has_header = detected["has_header"] if has_header is None else has_header
    frame = _read_preview_frame(
        file_path,
        selected_skip_rows,
        selected_delimiter,
        selected_has_header,
        max_preview_rows,
    )
    return FilePreview(
        path=file_path,
        file_size=file_path.stat().st_size,
        total_lines=_count_lines(file_path),
        delimiter=selected_delimiter,
        skip_rows=selected_skip_rows,
        has_header=selected_has_header,
        columns=[str(column) for column in frame.columns],
        preview_rows=frame.astype(str).values.tolist(),
        metadata=detected["metadata"],
    )


def _read_delimited_table(path: Path, options: DataImportOptions) -> pd.DataFrame:
    preview = preview_signal_file(
        path,
        skip_rows=options.skip_rows,
        delimiter=options.delimiter,
        has_header=options.has_header,
        max_preview_rows=1,
    )
    header = 0 if preview.has_header else None
    frame = pd.read_csv(
        path,
        sep=preview.delimiter,
        engine="python",
        header=header,
        skiprows=preview.skip_rows,
    )
    if header is None:
        frame.columns = [f"CH{i + 1}" for i in range(frame.shape[1])]
    return frame


def _sniff_delimiter(path: Path) -> str | None:
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ")
        return dialect.delimiter
    except csv.Error:
        return None


def _detect_table_layout(path: Path, delimiter: str | None) -> dict:
    entries = _read_sample_lines(path, 300)
    selected_delimiter = delimiter if delimiter is not None else _sniff_delimiter(path)
    metadata: dict[str, str] = {}
    data_row_index = 0
    header_row_index: int | None = None
    data_file_line = 0

    for idx, (line, file_line) in enumerate(entries):
        parts = _split_line(line, selected_delimiter)
        numeric_cells = _numeric_cell_count(parts)
        is_data_row = (
            len(parts) >= 2
            and numeric_cells >= max(2, len(parts) // 2)
            and _is_number(parts[0])
        )
        if is_data_row:
            data_row_index = idx
            data_file_line = file_line
            prev_entry = _previous_non_empty_entry(entries, idx)
            if prev_entry is not None:
                prev_line, prev_file_line = prev_entry
                prev_parts = _split_line(prev_line, selected_delimiter)
                if len(prev_parts) == len(parts) and _numeric_cell_count(prev_parts) == 0:
                    header_row_index = prev_file_line
            break
        if len(parts) >= 2 and parts[0]:
            metadata[parts[0]] = ",".join(parts[1:]).strip()
    else:
        raise ValueError("Could not find a numeric data table in the file.")

    return {
        "delimiter": selected_delimiter,
        "skip_rows": header_row_index if header_row_index is not None else data_file_line,
        "has_header": header_row_index is not None,
        "metadata": metadata,
    }


def _read_preview_frame(
    path: Path,
    skip_rows: int,
    delimiter: str | None,
    has_header: bool,
    max_preview_rows: int,
) -> pd.DataFrame:
    header = 0 if has_header else None
    frame = pd.read_csv(
        path,
        sep=delimiter,
        engine="python",
        header=header,
        skiprows=skip_rows,
        nrows=max_preview_rows,
    )
    if header is None:
        frame.columns = [f"CH{i + 1}" for i in range(frame.shape[1])]
    return frame


def _read_sample_lines(path: Path, limit: int) -> list[tuple[str, int]]:
    """Return (stripped_line, file_line_number) for non-empty lines."""
    entries: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line_num in range(limit):
            line = handle.readline()
            if line == "":
                break
            if line.strip():
                entries.append((line.strip(), line_num))
    if not entries:
        raise ValueError("file is empty.")
    return entries


def _split_line(line: str, delimiter: str | None) -> list[str]:
    if delimiter == r"\s+" or delimiter is None:
        if delimiter is None and "," in line:
            return [part.strip() for part in line.split(",")]
        if delimiter is None and "\t" in line:
            return [part.strip() for part in line.split("\t")]
        return [part.strip() for part in line.split()]
    return [part.strip() for part in line.split(delimiter)]


def _numeric_cell_count(parts: list[str]) -> int:
    return sum(1 for part in parts if _is_number(part))


def _previous_non_empty_entry(entries: list[tuple[str, int]], index: int) -> tuple[str, int] | None:
    for previous in range(index - 1, -1, -1):
        line, _ = entries[previous]
        if line.strip():
            return entries[previous]
    return None


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        return sum(1 for _ in handle)


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _frame_from_array(array: np.ndarray) -> pd.DataFrame:
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return pd.DataFrame({"CH1": array})
    if array.ndim == 2:
        return pd.DataFrame(array, columns=[f"CH{i + 1}" for i in range(array.shape[1])])
    raise ValueError("Only one- or two-dimensional NPY arrays are supported.")


def _frame_from_npz(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as data:
        if "time" in data and "values" in data:
            values = np.asarray(data["values"], dtype=float)
            frame = _frame_from_array(values)
            frame.insert(0, "time", np.asarray(data["time"], dtype=float))
            return frame
        if len(data.files) == 1:
            return _frame_from_array(data[data.files[0]])
    raise ValueError("NPZ must contain either time+values or a single array.")


def _frame_from_mat(path: Path) -> pd.DataFrame:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("scipy is required to read MAT files.") from exc

    raw = loadmat(path)
    arrays = {
        key: np.asarray(value).squeeze()
        for key, value in raw.items()
        if not key.startswith("__") and np.asarray(value).squeeze().ndim == 1
    }
    if not arrays:
        raise ValueError("MAT file does not contain one-dimensional numeric arrays.")
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("MAT arrays must share the same length.")
    return pd.DataFrame(arrays)


def _frame_to_signal(
    name: str,
    frame: pd.DataFrame,
    sample_rate: float | None,
    time_column: str | int | None = None,
    value_columns: list[str | int] | None = None,
) -> MultiChannelSignal:
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if numeric.empty:
        raise ValueError("No numeric signal columns were found.")

    time_column = _normalize_column_reference(time_column, numeric)
    if time_column is None:
        time_column = _detect_time_column(numeric)
    if time_column is not None:
        numeric = numeric[np.isfinite(numeric[time_column].to_numpy(dtype=float))]
        time = numeric[time_column].to_numpy(dtype=float)
        available_channels = numeric.drop(columns=[time_column])
        inferred_sample_rate = infer_sample_rate_from_time(time)
    else:
        if sample_rate is None or sample_rate <= 0:
            raise ValueError("sample_rate is required when no time column exists.")
        available_channels = numeric
        inferred_sample_rate = float(sample_rate)
        time = np.arange(len(available_channels), dtype=float) / inferred_sample_rate

    normalized_value_columns = _normalize_column_list(value_columns, available_channels)
    channels_frame = (
        available_channels[normalized_value_columns]
        if normalized_value_columns
        else available_channels
    )

    if channels_frame.empty:
        raise ValueError("No channel columns were found.")

    channels: dict[str, np.ndarray] = {}
    invalid_counts: dict[str, int] = {}
    for column in channels_frame.columns[:8]:
        values, invalid_count = _repair_channel_values(channels_frame[column])
        channels[str(column)] = values
        if invalid_count:
            invalid_counts[str(column)] = invalid_count

    return MultiChannelSignal(
        name=name,
        time=time,
        channels=channels,
        sample_rate=inferred_sample_rate,
        metadata={
            "channel_count": len(channels),
            "time_column": time_column,
            "repaired_nonfinite_points": invalid_counts,
        },
    )


def _repair_channel_values(series: pd.Series) -> tuple[np.ndarray, int]:
    """Return a finite channel vector, linearly filling isolated bad samples."""
    values = series.to_numpy(dtype=float)
    finite = np.isfinite(values)
    invalid_count = int(values.size - np.count_nonzero(finite))
    if invalid_count == 0:
        return values, 0
    if not np.any(finite):
        raise ValueError(f"Channel {series.name} contains no finite numeric samples.")
    if np.count_nonzero(finite) == 1:
        values = np.full_like(values, float(values[finite][0]))
        return values, invalid_count

    repaired = values.copy()
    x = np.arange(values.size, dtype=float)
    repaired[~finite] = np.interp(x[~finite], x[finite], values[finite])
    return repaired, invalid_count


def _detect_time_column(frame: pd.DataFrame) -> str | int | None:
    for column in frame.columns:
        if str(column).strip().lower() in TIME_COLUMN_HINTS:
            return column
    return None


def _normalize_column_reference(
    column: str | int | None, frame: pd.DataFrame
) -> str | int | None:
    if column is None or column == "":
        return None
    if column in frame.columns:
        return column
    column_text = str(column)
    for candidate in frame.columns:
        if str(candidate) == column_text:
            return candidate
    raise KeyError(f"Unknown column: {column}")


def _normalize_column_list(
    columns: list[str | int] | None, frame: pd.DataFrame
) -> list[str | int] | None:
    if not columns:
        return None
    return [_normalize_column_reference(column, frame) for column in columns]
