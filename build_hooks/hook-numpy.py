"""Custom hook to work around PyInstaller numpy hook issue on Python 3.13."""
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('numpy')
datas = collect_data_files('numpy')
