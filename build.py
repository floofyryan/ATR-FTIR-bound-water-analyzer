"""
Build script for FTIR Bound Water Analyzer
Run: python build.py
Output: dist/FTIR_Analyzer.exe  (Windows)
        dist/FTIR_Analyzer      (macOS/Linux)
"""
import subprocess
import sys

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--windowed",
    "--name", "FTIR_Analyzer",
    "--clean",
    "--add-data", "ftir_analyzer.py:.",   # not needed but shows pattern
    "--hidden-import", "scipy.optimize",
    "--hidden-import", "scipy.integrate",
    "--hidden-import", "scipy.special",
    "--hidden-import", "matplotlib.backends.backend_qtagg",
    "--hidden-import", "matplotlib.backends.backend_agg",
    "--hidden-import", "PyQt6.QtSvg",
    "--hidden-import", "PyQt6.QtPrintSupport",
    "ftir_analyzer.py"
]

print("Building FTIR Analyzer executable…")
result = subprocess.run(cmd, check=True)
print("\nBuild complete!  Find your executable in the  dist/  folder.")
