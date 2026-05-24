# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ftir_analyzer.py'],
    pathex=[],
    binaries=[],
    datas=[('ftir_analyzer.py', '.')],
    hiddenimports=['scipy.optimize', 'scipy.integrate', 'scipy.special', 'matplotlib.backends.backend_qtagg', 'matplotlib.backends.backend_agg', 'PyQt6.QtSvg', 'PyQt6.QtPrintSupport'],
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
