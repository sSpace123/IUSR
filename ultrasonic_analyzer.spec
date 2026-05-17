# -*- mode: python ; coding: utf-8 -*-

"""PyInstaller spec for ultrasonic signal analyzer."""

from pathlib import Path

block_cipher = None

anaconda_bin = Path("D:/Anaconda/Library/bin")
mkl_binaries = [
    (str(path), ".")
    for pattern in ("mkl*.dll", "libiomp5md.dll", "tbb*.dll")
    for path in anaconda_bin.glob(pattern)
]

hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtNetwork",
    "scipy.signal",
    "scipy.signal.windows",
    "scipy.io",
    "scipy.linalg",
    "scipy.sparse",
    "scipy.sparse.linalg",
    "pywt",
    "pywt._extensions._cwt",
    "pyqtgraph",
    "pyqtgraph.graphicsItems",
    "pyqtgraph.widgets",
    "pandas",
    "openpyxl",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=mkl_binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=["build_hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter", "matplotlib", "PIL", "jedi",
        "IPython", "jupyter", "notebook",
        "sphinx", "pytest", "setuptools", "pip",
        "PyQt5", "PyQt6", "PySide2", "qtpy",
        "dask", "numba", "llvmlite", "pyarrow",
        "tables", "sqlalchemy", "boto3", "botocore",
        "torch", "torchvision", "torchaudio", "tensorflow",
        "sklearn", "skimage", "cv2", "xarray",
        "distributed", "bokeh", "panel", "plotly",
        "altair", "statsmodels", "patsy", "intake",
        "h5py", "zarr", "fsspec", "nbformat",
        "nbconvert", "docutils", "markdown", "jupyter_client",
        "jupyter_core", "ipywidgets", "traitlets",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="UltrasonicSignalAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
