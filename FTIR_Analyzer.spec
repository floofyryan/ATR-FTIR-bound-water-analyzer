# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ftir_analyzer.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['scipy.optimize', 'scipy.optimize._lsq', 'scipy.optimize._minimize', 'scipy.integrate', 'scipy.integrate._tanhsinh', 'scipy.special', 'scipy.special._ufuncs', 'scipy.linalg', 'scipy._lib.messagestream', 'matplotlib.backends.backend_qtagg', 'matplotlib.backends.backend_agg', 'matplotlib.backends.backend_qt', 'PyQt6.QtSvg', 'PyQt6.QtSvgWidgets', 'PyQt6.QtPrintSupport', 'olefile', 'numpy.core._methods', 'numpy.lib.format'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FTIR_Analyzer',
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
)
