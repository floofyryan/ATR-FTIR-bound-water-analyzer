# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for FTIR Bound Water Analyzer.
# Run:  pyinstaller FTIR_Analyzer.spec   (or use build.py which passes the same flags)

a = Analysis(
    ['ftir_analyzer.py'],
    pathex=[],
    binaries=[],
    datas=[],    # no extra data files — everything is embedded in the single .py source
    hiddenimports=[
        # scipy — static analysis misses C-extension submodules
        'scipy.optimize',
        'scipy.optimize._lsq',
        'scipy.optimize._minimize',
        'scipy.integrate',
        'scipy.integrate._tanhsinh',
        'scipy.special',
        'scipy.special._ufuncs',
        'scipy.linalg',
        'scipy._lib.messagestream',
        # matplotlib Qt6 backend
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_agg',
        'matplotlib.backends.backend_qt',
        # PyQt6 optional plugins loaded at runtime
        'PyQt6.QtSvg',
        'PyQt6.QtSvgWidgets',
        'PyQt6.QtPrintSupport',
        # OLE parser for JASCO .jws binary spectrum files
        'olefile',
        # numpy internals
        'numpy.core._methods',
        'numpy.lib.format',
    ],
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
    console=False,                 # No console window — windowed GUI only
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
