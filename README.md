# Ultrasonic Signal Analyzer

Python MVP for quickly loading, viewing, filtering, analyzing, and exporting
ultrasonic nondestructive-testing signals.

## Current MVP

- Load CSV/TXT files, plus simple XLSX, NPY, NPZ, and MAT files when dependencies exist.
- Support single-channel and up to 8-channel data.
- Preview CSV/TXT imports, auto-detect metadata rows, and let users manually
  choose skipped rows, delimiters, time columns, and displayed signal columns.
- Use a three-column desktop workflow: data/channel management, tabbed analysis
  results, and a simplified operation panel with advanced filter settings hidden
  by default.
- Show multi-channel time traces, FFT spectra, CWT images, and a per-channel
  feature table from the main tab area.
- Auto-adjust invalid narrow-band and wavelet frequency ranges to stay below
  Nyquist, and align narrow-band results by the configured post-filter cycle
  count, following the MATLAB toneburst workflow.
- Optional OpenAI-compatible API assistant. It stays disabled by default; when
  enabled, users can provide Base URL, model, and API key to suggest import and
  analysis parameters.
- Hilbert envelope and TOF features are included in the feature table, along
  with energy, average power, envelope energy, and envelope area.
- Infer sample rate from a `time` column, or generate a time axis from a user-provided sample rate.
- Plot time-domain traces and FFT spectra in the PySide6 GUI.
- Run narrow-band band-pass filtering.
- Compute basic features: peak, peak-to-peak, RMS, energy, peak time, dominant frequency, and envelope peak.
- Compute CWT time-frequency coefficients.
- Estimate channel delay with cross-correlation in the core API.
- Export signal results to CSV.

## Install

Recommended on this machine: reuse the existing Anaconda scientific stack and
install only the small missing GUI helper package.

```powershell
D:\Anaconda\python.exe -m venv .venv-anaconda --system-site-packages
.\.venv-anaconda\Scripts\python.exe -m pip install pyqtgraph
```

For a fully independent environment, use:

```powershell
python -m pip install -r requirements.txt
```

## Run The GUI

```powershell
.\.venv-anaconda\Scripts\python.exe main.py
```

If GUI dependencies are missing, the core algorithms can still be tested and
used from scripts as long as `numpy` and `pandas` are available.

## Run Tests

```powershell
.\.venv-anaconda\Scripts\python.exe -m unittest discover -s tests
```

## Input Examples

Single channel:

```csv
time,signal
0.0000000,0.001
0.0000001,0.003
0.0000002,0.010
```

Multi-channel:

```csv
time,CH1,CH2,CH3
0.0000000,0.001,0.002,0.001
0.0000001,0.003,0.004,0.002
0.0000002,0.010,0.012,0.008
```

No time column:

```csv
CH1,CH2
0.001,0.002
0.003,0.004
0.010,0.012
```

For files without a time column, pass or enter the sample rate so the software
can generate the time axis.
