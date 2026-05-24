"""
Build script for FTIR Bound Water Analyzer.

Usage:
    python build.py

Output:
    dist/FTIR_Analyzer.exe   (Windows)
    dist/FTIR_Analyzer       (macOS / Linux)

The script drives PyInstaller with --onefile --windowed so that the result is a
single self-contained executable that launches without a console window.

Hidden imports are needed for modules that PyInstaller's static analysis misses
because they are loaded dynamically at runtime (e.g. matplotlib backends, scipy
submodules, and the olefile COM-document parser used to read JASCO .jws files).
"""
import subprocess
import sys

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--windowed",              # No console window on Windows / macOS
    "--name", "FTIR_Analyzer",
    "--clean",                 # Remove stale build artefacts before building

    # ── scipy submodules ──────────────────────────────────────────────────────
    # PyInstaller does not walk scipy's C-extension tree automatically.
    "--hidden-import", "scipy.optimize",
    "--hidden-import", "scipy.optimize._lsq",       # least_squares (TRF solver)
    "--hidden-import", "scipy.optimize._minimize",
    "--hidden-import", "scipy.integrate",
    "--hidden-import", "scipy.integrate._tanhsinh",
    "--hidden-import", "scipy.special",
    "--hidden-import", "scipy.special._ufuncs",
    "--hidden-import", "scipy.linalg",
    "--hidden-import", "scipy._lib.messagestream",

    # ── matplotlib backend ────────────────────────────────────────────────────
    "--hidden-import", "matplotlib.backends.backend_qtagg",
    "--hidden-import", "matplotlib.backends.backend_agg",
    "--hidden-import", "matplotlib.backends.backend_qt",

    # ── PyQt6 optional modules ────────────────────────────────────────────────
    # Qt6 loads SVG / print support at runtime for icon rendering and printing.
    "--hidden-import", "PyQt6.QtSvg",
    "--hidden-import", "PyQt6.QtSvgWidgets",
    "--hidden-import", "PyQt6.QtPrintSupport",

    # ── JASCO .jws binary file support ────────────────────────────────────────
    # .jws files are OLE Compound Document files; olefile reads their headers.
    "--hidden-import", "olefile",

    # ── numpy / pandas internals ──────────────────────────────────────────────
    "--hidden-import", "numpy.core._methods",
    "--hidden-import", "numpy.lib.format",

    # Entry point
    "ftir_analyzer.py",
]

print("Building FTIR Analyzer executable…")
print("Command:", " ".join(cmd))
result = subprocess.run(cmd, check=True)
print("\nBuild complete!  Find your executable in the  dist/  folder.")
