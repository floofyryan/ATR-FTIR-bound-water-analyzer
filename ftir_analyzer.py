# ─────────────────────────────────────────────────────────────────────────────
# FTIR Bound Water Analyzer
# Deconvolutes FTIR O-H stretch spectra (3000–3800 cm⁻¹) into free, intermediate,
# and bound water fractions using Gaussian fitting.  Replicates the original
# Colab notebook pipeline: SNIP baseline → smooth → normalize → subtract
# background → trim → second SNIP → Gaussian fit → area integration.
# ─────────────────────────────────────────────────────────────────────────────

import sys                          # Used for sys.argv (Qt app) and sys.exit
import numpy as np                  # Array math throughout — spectra are numpy arrays
from scipy.optimize import least_squares   # Levenberg-Marquardt Gaussian fitting (TRF method)
from scipy.integrate import trapezoid      # Trapezoidal integration for peak areas
import csv                          # Writing results to CSV files
import struct                       # Unpacking binary data from JASCO .jws files
from pathlib import Path            # Cross-platform file path handling

# Qt6 widget imports — every UI element used in the app
from PyQt6.QtWidgets import (
    QApplication,       # The top-level Qt application object
    QMainWindow,        # The main application window
    QWidget,            # Generic container widget
    QVBoxLayout,        # Stacks child widgets vertically
    QHBoxLayout,        # Places child widgets side by side horizontally
    QLabel,             # Displays text or images
    QPushButton,        # Clickable button
    QFileDialog,        # System file-picker dialog
    QDoubleSpinBox,     # Number input with decimal places (heights, sigmas)
    QSpinBox,           # Integer number input (SNIP iterations, smoothing window)
    QGroupBox,          # Labelled box that groups related controls
    QTabWidget,         # Container with switchable tabs (plot area)
    QStatusBar,         # Thin bar at window bottom showing status messages
    QScrollArea,        # Makes content scrollable — used for the left control panel
    QFrame,             # Generic frame; used for divider lines and styled cards
    QSizePolicy,        # Controls how widgets expand/shrink to fill space
    QMessageBox,        # Pop-up alert dialog (errors, warnings)
    QDialog,            # Base class for the paste-data modal dialog
    QTextEdit,          # Multi-line text box — used in PasteDialog for pasted data
    QDialogButtonBox,   # OK / Cancel button row at the bottom of a dialog
    QListWidget,        # Scrollable list — used in batch mode for sample files
    QListWidgetItem,    # A single row inside a QListWidget
    QProgressBar,       # Progress bar shown during batch processing
    QSlider,            # Draggable slider for the normalization wavenumber
    QCheckBox,          # Tickbox — "Raise minimum to zero" option in Step 1
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
# Qt:          namespace for alignment flags, orientation, etc.
# QThread:     base class for background worker threads (fitting, processing)
# pyqtSignal:  decorator to declare custom signals on QThread subclasses

from PyQt6.QtGui import QFont, QColor   # QFont: font objects; QColor: list-item text colors

import matplotlib
matplotlib.use("QtAgg")             # Tell matplotlib to render into Qt6 widgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# FigureCanvas: bridges a matplotlib Figure into a Qt widget that can be embedded
from matplotlib.figure import Figure  # The matplotlib drawing surface

# ── Palette ───────────────────────────────────────────────────────────────────
# All colours are defined here so they can be referenced throughout the
# stylesheet and Python code from a single source of truth.
BG        = "#0f1117"   # Very dark navy — main window background
SURFACE   = "#1a1d27"   # Slightly lighter — card / panel backgrounds
SURFACE2  = "#222536"   # Even lighter — input field backgrounds
BORDER    = "#2e3148"   # Subtle border colour between elements
ACCENT    = "#5b6af0"   # Indigo — primary interactive colour (sliders, active steps)
ACCENT2   = "#7c8bff"   # Lighter indigo — hover states and secondary highlights
TEXT      = "#e8eaf6"   # Near-white — primary readable text
TEXT_DIM  = "#7b82a8"   # Muted purple-grey — hints, labels, secondary text
FREE_CLR  = "#f06292"   # Pink-red — free water peak colour
INTER_CLR = "#4dd0a0"   # Teal-green — intermediate water peak colour
BOUND_CLR = "#5b8def"   # Blue — bound water peak colour
SUCCESS   = "#4dd0a0"   # Green — used on "done" step cards and loaded file labels
DANGER    = "#f06292"   # Red — used for warnings and errors
STEP_DONE = "#2e3f2e"   # Dark green tint — background of a completed step card
STEP_ACT  = "#1e2540"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px 14px 14px 14px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    color: {TEXT_DIM};
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px; top: 2px;
    padding: 0 6px;
    background-color: {SURFACE};
}}
QLabel {{ color: {TEXT}; background: transparent; }}
QDoubleSpinBox, QSpinBox {{
    background-color: {SURFACE2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    color: {TEXT};
    min-width: 90px;
    font-size: 13px;
}}
QDoubleSpinBox:focus, QSpinBox:focus {{ border: 1px solid {ACCENT}; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {{
    background: {SURFACE2}; border: none; width: 18px;
}}
QPushButton {{
    background-color: {SURFACE2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 500;
    min-height: 36px;
    min-width: 80px;
}}
QPushButton:hover {{ background-color: {BORDER}; border-color: {ACCENT}; }}
QPushButton:pressed {{ background-color: {ACCENT}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {BORDER}; }}
QPushButton#accent {{
    background-color: {ACCENT};
    color: white; border: none;
    font-size: 14px; font-weight: 600;
    padding: 12px 28px;
    min-height: 42px;
    border-radius: 10px;
}}
QPushButton#accent:hover {{ background-color: {ACCENT2}; }}
QPushButton#accent:disabled {{ background-color: {SURFACE2}; color: {TEXT_DIM}; }}
QPushButton#save {{
    background-color: {SURFACE2};
    color: {INTER_CLR};
    border: 1px solid {INTER_CLR};
    border-radius: 8px;
    padding: 9px 18px;
    font-size: 13px;
    min-height: 36px;
}}
QPushButton#save:hover {{ background-color: #1a2e28; }}
QPushButton#save:disabled {{ color: {TEXT_DIM}; border-color: {BORDER}; }}
QPushButton#toggle {{
    background-color: {SURFACE2};
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 9px 16px;
    font-size: 13px;
    min-height: 36px;
}}
QPushButton#toggle:checked {{
    background-color: {ACCENT};
    color: white;
    border-color: {ACCENT};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {SURFACE};
    top: -1px;
}}
QTabBar::tab {{
    background: {SURFACE2}; color: {TEXT_DIM};
    border: 1px solid {BORDER}; border-bottom: none;
    border-radius: 6px 6px 0 0;
    padding: 7px 18px; margin-right: 3px; font-size: 12px;
}}
QTabBar::tab:selected {{ background: {SURFACE}; color: {TEXT}; border-bottom-color: {SURFACE}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {SURFACE}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 4px; min-height: 24px;
}}
QStatusBar {{
    background: {SURFACE}; color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
    font-size: 12px; padding: 2px 8px;
}}
QFrame#divider {{ background: {BORDER}; max-height: 1px; }}
QFrame#step_card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QFrame#step_card[active=true] {{
    border-color: {ACCENT};
    background: {STEP_ACT};
}}
QFrame#step_card[done=true] {{
    border-color: {SUCCESS};
    background: {STEP_DONE};
}}
QTextEdit {{
    background-color: {SURFACE2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 6px;
}}
QTextEdit:focus {{ border: 1px solid {ACCENT}; }}
QDialog {{ background-color: {BG}; }}
QListWidget {{
    background-color: {SURFACE2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT};
    font-size: 12px;
}}
QListWidget::item {{ padding: 4px 8px; }}
QListWidget::item:selected {{ background: {ACCENT}; color: white; }}
QProgressBar {{
    background: {SURFACE2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    height: 10px;
    text-align: center;
    font-size: 10px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QSlider::groove:horizontal {{
    background: {BORDER};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    border: 2px solid {ACCENT2};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 9px;
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT2};
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
# Small utility functions shared across the UI-building code.

def make_spinbox(min_val, max_val, value, decimals=0, step=1.0):
    """Create either a QSpinBox (integer) or QDoubleSpinBox (decimal)
    depending on whether decimals > 0, pre-configured with range and value."""
    if decimals > 0:
        w = QDoubleSpinBox()          # Decimal input box (e.g. height = 0.100)
        w.setDecimals(decimals)       # How many decimal places to display
        w.setSingleStep(step)         # How much one arrow-click changes the value
    else:
        w = QSpinBox()                # Integer input box (e.g. SNIP iterations = 200)
        w.setSingleStep(int(step))    # Step must be an integer for QSpinBox
    w.setRange(min_val, max_val)      # Clamp the allowed input range
    w.setValue(value)                 # Set the starting value
    return w

def param_row(label_text, widget, hint=None):
    """Build a horizontal row: [label] [optional hint] [stretch] [widget].
    Used to lay out every parameter control in the step panels."""
    row = QHBoxLayout()
    lbl = QLabel(label_text)                          # Main parameter name
    lbl.setStyleSheet(f"color:{TEXT};font-size:12px;")
    row.addWidget(lbl)
    if hint:                                           # Optional grey hint text
        hl = QLabel(hint)
        hl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        row.addWidget(hl)
    row.addStretch()                                   # Push the input widget to the right
    row.addWidget(widget)                              # The actual spinbox or control
    return row

def hline():
    """Return a thin horizontal divider line for visual separation between sections."""
    f = QFrame(); f.setObjectName("divider")          # "divider" maps to the CSS rule
    f.setFrameShape(QFrame.Shape.HLine)               # Draw as a horizontal line
    return f

def save_csv_two_col(path, x, y, col1="wavenumber", col2="absorbance"):
    """Write two parallel arrays to a two-column CSV file.
    Used to save intermediate spectrum data (normalized, subtracted, trimmed)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([col1, col2])                      # Header row
        for xi, yi in zip(x, y):
            w.writerow([f"{xi:.4f}", f"{yi:.6f}"])    # One data point per row

# ── Science (unchanged) ───────────────────────────────────────────────────────

def _parse_jws(path_bytes):
    """Read a JASCO .jws binary (OLE2) file.
    Returns xs, ys (np.arrays), metadata (dict).
    Requires the olefile package (pip install olefile).
    """
    try:
        import olefile
        import io
        ole = olefile.OleFileIO(io.BytesIO(path_bytes))
        di = ole.openstream('DataInfo').read()
        npoints  = struct.unpack_from('<I', di, 20)[0]
        x_start  = struct.unpack_from('<d', di, 24)[0]
        x_end    = struct.unpack_from('<d', di, 32)[0]
        y_raw    = ole.openstream('Y-Data').read()
        y        = np.frombuffer(y_raw, dtype='<f4').astype(float)[:npoints]
        x        = np.linspace(x_start, x_end, npoints)
        meta = {"FORMAT": "JASCO JWS", "NPOINTS": str(npoints),
                "FIRSTX": f"{x_start:.4f}", "LASTX": f"{x_end:.4f}"}
        try:
            ui = ole.openstream('UserInfo').read().decode('latin-1', errors='ignore')
            ui = ''.join(c for c in ui if c.isprintable()).strip()
            if ui: meta["USER"] = ui
        except Exception:
            pass
        return x, y, meta
    except ImportError:
        raise RuntimeError(
            "Reading .jws files requires the olefile package.\n"
            "Run:  pip install olefile")


def parse_spectrum(text_or_bytes, raw_bytes=None):
    """Parse FTIR spectrum files robustly.

    Handles:
    - JASCO .jws binary files (OLE2 format) — pass raw_bytes=<bytes>
    - JASCO text exports with metadata header and XYDATA marker
    - Plain two-column CSV / whitespace files (comma, tab, space delimited)
    - Pasted text with the same two-column structure
    - Any file where header/footer lines are non-numeric (skipped automatically)

    Returns: xs (np.array), ys (np.array), metadata (dict)
    """
    # Binary JWS path
    if raw_bytes is not None and raw_bytes[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        return _parse_jws(raw_bytes)

    # Text path
    if isinstance(text_or_bytes, bytes):
        for enc in ('utf-8', 'latin-1', 'utf-16-le'):
            try:
                text = text_or_bytes.decode(enc)
                break
            except Exception:
                text = text_or_bytes.decode('latin-1', errors='replace')
    else:
        text = text_or_bytes

    lines = text.strip().splitlines()
    metadata = {}
    xs, ys = [], []

    # Look for XYDATA marker (JASCO text exports and similar instrument formats)
    data_start = None
    for i, line in enumerate(lines):
        clean = line.strip().replace("\r", "")
        if clean.upper() == "XYDATA":
            data_start = i + 1
            break
        # Harvest metadata key,value pairs from header
        parts = clean.split(",", 1)
        if len(parts) == 2:
            k, v = parts[0].strip(), parts[1].strip()
            if k and not k.replace(".", "").replace("-", "").replace(" ", "").isdigit():
                metadata[k] = v

    scan_lines = lines[data_start:] if data_start is not None else lines

    for line in scan_lines:
        clean = line.strip().replace("\r", "")
        if not clean:
            continue
        # Accept comma, tab, semicolon, or whitespace delimiters
        parts = clean.replace(",", " ").replace("\t", " ").replace(";", " ").split()
        if len(parts) < 2:
            if xs: break   # footer after data
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
            xs.append(x); ys.append(y)
        except ValueError:
            if xs: break   # non-numeric footer
            continue       # non-numeric header line

    return np.array(xs), np.array(ys), metadata

def rolling_mean(y, w):
    """Smooth a spectrum with a simple box (rolling average) filter.
    Replaces each point with the mean of its w nearest neighbours.
    Reduces high-frequency noise without distorting peak shapes."""
    w = max(1, int(w))                              # Ensure window is at least 1 point
    return np.convolve(y, np.ones(w) / w, mode="same")
    # np.ones(w)/w is a flat kernel of width w;  mode="same" keeps output length equal to input

def snip_baseline(y, iterations):
    """Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP) baseline correction.

    Algorithm (Morháč 1997):
      1. Transform y into a log-log-sqrt space to compress peak amplitudes.
         This makes peaks look small relative to the slowly-varying baseline.
      2. Iteratively replace each interior point with the minimum of itself and
         the average of its two neighbours at distance m (m grows from 1 to iterations).
         This progressively erodes the transformed baseline estimate downward.
      3. Inverse-transform back to absorbance units.
      4. Subtract the estimated baseline from the original spectrum.

    The result is a spectrum where the background has been removed, leaving
    only the peaks riding on a near-zero baseline.
    """
    # Step 1: log-log-sqrt transform — np.maximum(y,0) guards against negative values
    # before sqrt, which would produce NaN.
    S = np.log(np.log(np.sqrt(np.maximum(y, 0) + 1) + 1) + 1)
    f = S.copy()                                    # f will become the baseline estimate
    for m in range(1, int(iterations) + 1):
        # Start at m=1 (not 0) — m=0 would make tmp[0:-0] = tmp[0:0] = empty slice
        tmp = f.copy()                              # Work on a copy so current iteration is consistent
        # Replace each point with min(itself, average of neighbours m steps away)
        # Only update the interior [m:-m] to avoid boundary index errors
        tmp[m:-m] = np.minimum(f[m:-m], (f[:len(f)-2*m] + f[2*m:]) / 2)
        f = tmp                                     # Keep the clipped version for next iteration
    # Step 3: inverse transform — undo the log-log-sqrt to get back to absorbance units
    S_prime = (np.exp(np.exp(f) - 1) - 1) ** 2 - 1
    return y - S_prime                              # Step 4: subtract estimated baseline

def gaus(x, a, x0, sigma):
    """Evaluate a single Gaussian curve at positions x.
    a     = peak height (amplitude)
    x0    = peak centre (wavenumber)
    sigma = peak width parameter (standard deviation, related to FWHM by 2.355*sigma)
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def process_spectrum(xs, ys, snip_iter, smooth_win, norm_wav):
    """Apply the full preprocessing pipeline to one spectrum:
      1. Ensure wavenumbers are ascending (some instruments export descending).
      2. SNIP baseline correction.
      3. Rolling-mean smoothing.
      4. Normalise: divide every point by the absorbance at norm_wav (e.g. 1740 cm⁻¹ C=O peak).
         This makes the background and sample spectra comparable before subtraction.
    """
    if xs[0] > xs[-1]:
        xs, ys = xs[::-1], ys[::-1]                # Reverse so wavenumbers run low→high
    bl = snip_baseline(ys, snip_iter)              # Remove slowly varying fluorescence/scattering background
    sm = rolling_mean(bl, smooth_win)              # Smooth out high-frequency detector noise
    norm_idx = np.argmin(np.abs(xs - norm_wav))    # Find the index closest to the reference wavenumber
    nv = sm[norm_idx] if sm[norm_idx] != 0 else 1.0   # Guard against divide-by-zero
    return xs, sm / nv                              # Return ascending x and normalised y

def _build_bounds(anchor_mode, fit_mode, params, constraints):
    """Construct the (lower_bounds, upper_bounds) arrays required by least_squares.

    Anchored mode: centres are NOT in the parameter vector, so only heights and
    sigmas need bounds.  Heights are bounded ≥ 0 (physically: absorbance can't
    be negative).  Sigmas are bounded to [sig_min, sig_max] to prevent the
    optimizer producing meaninglessly narrow (spike) or wide (flat) peaks.

    Floating mode: centres ARE in the parameter vector and are bounded to
    ±cw (80) cm⁻¹ around their initial guesses.  This prevents peaks from
    swapping order or drifting to physically unreasonable wavenumbers.
    """
    sig_lo, sig_hi, INF = constraints["sig_min"], constraints["sig_max"], np.inf
    fc, ic, bc = params["fc"], params["ic"], params["bc"]   # Initial centre guesses
    cw = 80                    # Maximum allowed centre shift in cm⁻¹ (floating mode)
    if fit_mode == "triple":
        if anchor_mode == "anchored":
            # Parameter order: [h_free, sig_free, h_inter, sig_inter, h_bound, sig_bound]
            return ([0,sig_lo,0,sig_lo,0,sig_lo], [INF,sig_hi,INF,sig_hi,INF,sig_hi])
        else:
            # Parameter order: [h_f, c_f, s_f, h_i, c_i, s_i, h_b, c_b, s_b]
            return ([0,fc-cw,sig_lo,0,ic-cw,sig_lo,0,bc-cw,sig_lo],
                    [INF,fc+cw,sig_hi,INF,ic+cw,sig_hi,INF,bc+cw,sig_hi])
    else:                      # double Gaussian — no intermediate peak
        if anchor_mode == "anchored":
            # Parameter order: [h_free, sig_free, h_bound, sig_bound]
            return ([0,sig_lo,0,sig_lo], [INF,sig_hi,INF,sig_hi])
        else:
            # Parameter order: [h_f, c_f, s_f, h_b, c_b, s_b]
            return ([0,fc-cw,sig_lo,0,bc-cw,sig_lo],
                    [INF,fc+cw,sig_hi,INF,bc+cw,sig_hi])

def run_fit(xT, yT, params, fit_mode, anchor_mode, constraints):
    """Fit two or three Gaussians to the processed O-H stretch spectrum.

    Parameters
    ----------
    xT          : wavenumber array for the trimmed region (ascending)
    yT          : absorbance array after SNIP + subtraction
    params      : dict of initial guesses {fc, fh, fs, ic, ih, isg, bc, bh, bs}
                  (centre, height, sigma for free / intermediate / bound)
    fit_mode    : "triple" (3 Gaussians) or "double" (2 Gaussians, no intermediate)
    anchor_mode : "anchored" — centres fixed, only height+sigma optimised
                  "float"    — all 9 (or 6) parameters optimised
    constraints : dict with sig_min and sig_max to bound sigma during fitting

    Returns a dict with: spectral arrays, areas, percentages, R², RMSE,
    fitted parameters, centre positions, and any quality warnings.
    """
    # Unpack initial guess values from the params dict
    fc,fh,fs = params["fc"],params["fh"],params["fs"]   # Free: centre, height, sigma
    ic,ih,isg = params["ic"],params["ih"],params["isg"] # Intermediate
    bc,bh,bs = params["bc"],params["bh"],params["bs"]   # Bound
    bounds = _build_bounds(anchor_mode, fit_mode, params, constraints)  # Build (lo, hi) arrays
    # ── Build model functions and run the optimiser ───────────────────────────
    if fit_mode == "triple":
        if anchor_mode == "anchored":
            # Anchored: centres are NOT in the parameter vector — they are
            # captured from outer scope (fc, ic, bc) and held fixed.
            # p = [h_free, sig_free, h_inter, sig_inter, h_bound, sig_bound]
            def model(p,x): return gaus(x,p[0],fc,p[1])+gaus(x,p[2],ic,p[3])+gaus(x,p[4],bc,p[5])
            res = least_squares(lambda p: model(p,xT)-yT, [fh,fs,ih,isg,bh,bs],
                                method="trf", bounds=bounds, max_nfev=4000)
            # Rebuild the full 9-element fp list by inserting fixed centres
            p=res.x; fp=[p[0],fc,p[1],p[2],ic,p[3],p[4],bc,p[5]]
        else:
            # Floating: all 9 parameters are optimised
            # p = [h_f, c_f, s_f, h_i, c_i, s_i, h_b, c_b, s_b]
            def model(p,x): return gaus(x,p[0],p[1],p[2])+gaus(x,p[3],p[4],p[5])+gaus(x,p[6],p[7],p[8])
            res = least_squares(lambda p: model(p,xT)-yT, [fh,fc,fs,ih,ic,isg,bh,bc,bs],
                                method="trf", bounds=bounds, max_nfev=4000)
            fp=res.x.tolist()   # Already has 9 elements
        # Evaluate each individual Gaussian curve at the fitted parameters
        yfit=gaus(xT,fp[0],fp[1],fp[2])+gaus(xT,fp[3],fp[4],fp[5])+gaus(xT,fp[6],fp[7],fp[8])
        fc_=gaus(xT,fp[0],fp[1],fp[2])  # Free water Gaussian curve
        ic_=gaus(xT,fp[3],fp[4],fp[5])  # Intermediate water Gaussian curve
        bc_=gaus(xT,fp[6],fp[7],fp[8])  # Bound water Gaussian curve
        # Trapezoidal integration gives the area under each peak
        fA=trapezoid(fc_,xT); iA=trapezoid(ic_,xT); bA=trapezoid(bc_,xT)
        centers={"free":fp[1],"inter":fp[4],"bound":fp[7]}  # Fitted centre positions
    else:   # double Gaussian
        if anchor_mode == "anchored":
            # p = [h_free, sig_free, h_bound, sig_bound]
            def model(p,x): return gaus(x,p[0],fc,p[1])+gaus(x,p[2],bc,p[3])
            res = least_squares(lambda p: model(p,xT)-yT, [fh,fs,bh,bs],
                                method="trf", bounds=bounds, max_nfev=4000)
            p=res.x; fp=[p[0],fc,p[1],p[2],bc,p[3]]
        else:
            # p = [h_f, c_f, s_f, h_b, c_b, s_b]
            def model(p,x): return gaus(x,p[0],p[1],p[2])+gaus(x,p[3],p[4],p[5])
            res = least_squares(lambda p: model(p,xT)-yT, [fh,fc,fs,bh,bc,bs],
                                method="trf", bounds=bounds, max_nfev=4000)
            fp=res.x.tolist()
        yfit=gaus(xT,fp[0],fp[1],fp[2])+gaus(xT,fp[3],fp[4],fp[5])
        fc_=gaus(xT,fp[0],fp[1],fp[2])   # Free water curve
        ic_=np.zeros_like(xT)             # No intermediate peak — zero array
        bc_=gaus(xT,fp[3],fp[4],fp[5])   # Bound water curve
        fA=trapezoid(fc_,xT); iA=0.0; bA=trapezoid(bc_,xT)
        centers={"free":fp[1],"inter":None,"bound":fp[4]}  # inter=None for double
    # ── Compute fit quality metrics ───────────────────────────────────────────
    total=fA+iA+bA                                       # Total integrated area
    ss_res=np.sum((yT-yfit)**2)                          # Sum of squared residuals
    ss_tot=np.sum((yT-np.mean(yT))**2)                   # Total sum of squares
    r2=1-ss_res/ss_tot if ss_tot else 0                  # R² coefficient of determination
    rmse=np.sqrt(ss_res/len(yT))                         # Root mean squared error in AU
    # ── Fit quality warnings ──────────────────────────────────────────────────
    warns=[]
    # Check for suspiciously narrow peaks (sigma < 10 cm⁻¹ is physically unrealistic)
    sigs=([fp[2],fp[5],fp[8]],["free","intermediate","bound"]) if fit_mode=="triple" else ([fp[2],fp[5]],["free","bound"])
    for nm,sg in zip(sigs[1],sigs[0]):
        if abs(sg)<10: warns.append(f"{nm} sigma ({abs(sg):.1f} cm⁻¹) very narrow.")
    if r2<0.95: warns.append(f"R²={r2:.4f} below 0.95 — fit may be poor.")
    heights_idx=[0]+([3,6] if fit_mode=="triple" else [3])  # Indices of height params in fp
    if any(fp[i]<0 for i in heights_idx): warns.append("Negative height — physically unreasonable.")
    # ── Return all results as a single dict ───────────────────────────────────
    return {
        "xT":      xT,          # Wavenumber array (trimmed region)
        "yT":      yT,          # Absorbance data (trimmed, baseline-corrected)
        "yfit":    yfit,        # Total fitted curve (sum of all Gaussians)
        "free_curve":  fc_,     # Individual free water Gaussian curve
        "inter_curve": ic_,     # Individual intermediate water Gaussian curve
        "bound_curve": bc_,     # Individual bound water Gaussian curve
        "free_a":  abs(fA),     # Area under free peak (abs() guards descending-x edge case)
        "inter_a": abs(iA),     # Area under intermediate peak
        "bound_a": abs(bA),     # Area under bound peak
        "free_pct":  fA/total if total else 0,   # Fraction of total area that is free
        "inter_pct": iA/total if total else 0,   # Fraction that is intermediate
        "bound_pct": bA/total if total else 0,   # Fraction that is bound
        "r2":   r2,             # Coefficient of determination
        "rmse": rmse,           # Root mean squared error in absorbance units
        "centers":  centers,    # Dict of fitted peak centre positions
        "fp":       fp,         # Full 9- (or 6-) element fitted parameter vector
        "fit_mode": fit_mode,   # "triple" or "double"
        "warnings": warns,      # List of quality-control warning strings
    }

# ── Batch worker ─────────────────────────────────────────────────────────────

class BatchWorker(QThread):
    """Processes multiple sample files against one background sequentially.
    Emits progress(index, name, result_or_error_str) after each file.
    Emits finished(list_of_results) when done.
    """
    progress = pyqtSignal(int, str, object)   # (file_index, filename, result_dict_or_error)
    finished = pyqtSignal(list)               # Emits list of all results when done

    def __init__(self, bg_raw, sample_paths, settings):
        super().__init__()                       # Initialise QThread
        self.bg_raw       = bg_raw              # Background (xs, ys) arrays
        self.sample_paths = sample_paths         # List of (name, xs, ys) tuples
        self.s            = settings             # Full preprocessing + fit settings dict

    def run(self):
        """Called automatically by QThread.start() on a worker thread."""
        s = self.s         # Settings shorthand
        results = []       # Accumulate result dicts (or error dicts) here
        # Process the background ONCE — shared by all sample files
        try:
            bx, by = process_spectrum(*self.bg_raw, s["snip"], s["smooth"], s["norm_wav"])
        except Exception as e:
            # If background fails, every sample fails — emit error for each and quit
            for i, (name, xs, ys) in enumerate(self.sample_paths):
                self.progress.emit(i, name, f"Background error: {e}")
            self.finished.emit([])
            return

        for i, (name, xs, ys) in enumerate(self.sample_paths):
            try:
                # Normalise sample using same settings as the background
                sx, sy = process_spectrum(xs, ys, s["snip"], s["smooth"], s["norm_wav"])
                # Resample background to sample grid and subtract
                by_r   = np.interp(sx, bx, by)    # Linear interpolation onto sample x-grid
                sub    = sy - by_r                 # Background-subtracted spectrum
                # Find trim region boundaries
                ti  = np.argmin(np.abs(sx - s["trim_s"]))  # Index nearest trim start
                tei = np.argmin(np.abs(sx - s["trim_e"]))  # Index nearest trim end
                lo, hi = sorted([ti, tei])                  # Ensure lo < hi
                xT = sx[lo:hi+1]                            # Trimmed x array
                # Second SNIP pass on the O-H stretch region only
                yT = snip_baseline(sub[lo:hi+1], min(s["snip"], 100))
                # Run the Gaussian fit
                r  = run_fit(xT, yT, s["params"], s["fit_mode"], s["anchor_mode"], s["constraints"])
                r["sample_name"] = name    # Tag result with filename for the CSV
                results.append(r)
                self.progress.emit(i, name, r)   # Update UI with this file's result
            except Exception as e:
                # File failed — record error but continue with remaining files
                self.progress.emit(i, name, str(e))
                results.append({"sample_name": name, "error": str(e)})
        self.finished.emit(results)   # All done — UI can enable export


# ── Worker thread ─────────────────────────────────────────────────────────────

class StepWorker(QThread):
    """Generic one-shot background worker.  Accepts any callable and runs it
    on a background thread so the Qt event loop (and the UI) stays responsive
    during expensive operations like SNIP baseline or Gaussian fitting.

    Usage:
        worker = StepWorker(my_function, arg1, arg2)
        worker.finished.connect(my_callback)
        worker.start()
    """
    finished = pyqtSignal(object)   # Emitted with the return value of fn()
    error    = pyqtSignal(str)      # Emitted with the exception message if fn() raises

    def __init__(self, fn, *args, **kwargs):
        super().__init__()          # Initialise QThread
        self._fn = fn               # The function to run on the worker thread
        self._args = args           # Positional arguments to pass to fn
        self._kwargs = kwargs       # Keyword arguments to pass to fn

    def run(self):
        """Called by QThread.start().  Executes fn and emits result or error."""
        try:
            self.finished.emit(self._fn(*self._args, **self._kwargs))
        except Exception as e:
            import traceback; traceback.print_exc()   # Print full traceback to console
            self.error.emit(str(e))                   # Send error message to UI

# ── Plot canvas ───────────────────────────────────────────────────────────────

class PlotCanvas(FigureCanvas):
    """A matplotlib Figure embedded in a Qt widget.
    All plots in the app share this class.  Each plot method clears the figure,
    creates a styled axes, draws the data, and calls self.draw() to trigger a repaint.
    Using a single canvas class keeps styling consistent and avoids code duplication.
    """
    def __init__(self):
        # Create the matplotlib figure with the dark background colour
        self.fig = Figure(figsize=(6, 3.8), dpi=110, facecolor=SURFACE)
        super().__init__(self.fig)   # Initialise FigureCanvas (connects fig to Qt)
        # Allow the canvas to expand in both directions to fill available space
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _ax(self):
        """Create a new subplot with the app's dark theme applied.
        Called at the start of every plot method — always clears and recreates
        the axes so each plot is drawn fresh without leftover data."""
        ax = self.fig.add_subplot(111)           # Single plot filling the figure
        ax.set_facecolor(SURFACE)                # Dark background inside the plot area
        ax.tick_params(colors=TEXT_DIM, labelsize=9)   # Dim tick labels
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)   # Subtle axis borders
        ax.xaxis.label.set_color(TEXT_DIM); ax.yaxis.label.set_color(TEXT_DIM)  # Dim axis titles
        ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6, linestyle="--")   # Faint grid
        return ax

    def _draw(self, fn):
        """Clear the figure, create a styled axes, run fn(ax) to draw content,
        then tighten layout and trigger a Qt repaint."""
        self.fig.clear()           # Remove all previous content
        ax = self._ax()            # Create fresh themed axes
        fn(ax)                     # Let the caller draw into ax
        self.fig.tight_layout(pad=1.2)   # Automatically adjust margins
        self.draw()                # Trigger Qt to repaint the widget

    def plot_raw(self, bg, samp, norm_wav=None):
        sx,sy=samp
        if sx[0]>sx[-1]: sx,sy=sx[::-1],sy[::-1]
        if bg is not None:
            bx,by=bg
            if bx[0]>bx[-1]: bx,by=bx[::-1],by[::-1]
        def fn(ax):
            if bg is not None:
                ax.plot(bx,by,color=TEXT_DIM,lw=1.2,label="Background",alpha=0.85)
            ax.plot(sx,sy,color=BOUND_CLR,lw=1.4,label="Sample")
            if norm_wav is not None:
                ax.axvline(norm_wav,color=ACCENT2,lw=1.5,linestyle="--",alpha=0.9,
                           label=f"Norm. @ {norm_wav} cm⁻¹")
            ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Absorbance (AU)")
            ax.invert_xaxis()
            ax.legend(facecolor=SURFACE2,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
        self._draw(fn)

    def plot_processed(self, bg, samp, norm_wav=None):
        sx,sy = samp
        bx,by = bg if bg is not None else (None, None)
        def fn(ax):
            if bg is not None:
                ax.plot(bx,by,color=TEXT_DIM,lw=1.1,label="Background (norm.)",alpha=0.7)
            ax.plot(sx,sy,color=ACCENT2,lw=1.4,label="Sample (norm.)")
            if norm_wav is not None:
                ax.axvline(norm_wav,color=FREE_CLR,lw=1.5,linestyle="--",alpha=0.9,
                           label=f"Norm. @ {norm_wav} cm⁻¹")
                # Mark the normalisation point — curves should equal ~1 here
                import numpy as np
                marks = [(sx,sy,ACCENT2)]
                if bg is not None:
                    marks.insert(0, (bx,by,TEXT_DIM))
                for xs,ys,col in marks:
                    idx = int(np.argmin(np.abs(xs - norm_wav)))
                    ax.plot(xs[idx], ys[idx], "o", color=col, ms=6, zorder=5)
            ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Norm. Absorbance")
            ax.invert_xaxis()
            ax.legend(facecolor=SURFACE2,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
        self._draw(fn)

    def plot_subtracted(self, sx, sub, trim_s=None, trim_e=None, no_bg=False):
        curve_lbl = "Sample (normalized)" if no_bg else "Sample − Background"
        y_lbl     = "Absorbance (AU)" if no_bg else "ΔAbsorbance"
        def fn(ax):
            ax.plot(sx,sub,color=FREE_CLR,lw=1.4,label=curve_lbl)
            ax.axhline(0,color=BORDER,lw=0.8,linestyle="--")
            if trim_s is not None and trim_e is not None:
                lo, hi = min(trim_s,trim_e), max(trim_s,trim_e)
                ax.axvspan(lo, hi, alpha=0.13, color=INTER_CLR, label=f"Trim region ({lo}–{hi} cm⁻¹)")
                ax.axvline(lo, color=INTER_CLR, lw=1.2, linestyle="--", alpha=0.8)
                ax.axvline(hi, color=INTER_CLR, lw=1.2, linestyle="--", alpha=0.8)
            ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel(y_lbl)
            ax.invert_xaxis()
            ax.legend(facecolor=SURFACE2,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
        self._draw(fn)

    def plot_trimmed(self, xT, yT, trim_s, trim_e):
        def fn(ax):
            ax.plot(xT,yT,color=INTER_CLR,lw=1.5,label=f"Trimmed & corrected ({trim_s}–{trim_e} cm⁻¹)")
            ax.axhline(0,color=BORDER,lw=0.8,linestyle="--")
            ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Absorbance (AU)")
            ax.invert_xaxis()
            ax.legend(facecolor=SURFACE2,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
        self._draw(fn)

    def plot_fit(self, r):
        xT=r["xT"]
        def fn(ax):
            ax.plot(xT,r["yT"],color=TEXT,lw=1.6,label="Data",alpha=0.9)
            ax.plot(xT,r["yfit"],color="white",lw=1.8,linestyle="--",label="Total fit",alpha=0.85)
            ax.fill_between(xT,r["free_curve"],alpha=0.2,color=FREE_CLR)
            ax.plot(xT,r["free_curve"],color=FREE_CLR,lw=1.4,
                    label=f"Free ({r['free_pct']*100:.1f}%)")
            if r["fit_mode"]=="triple":
                ax.fill_between(xT,r["inter_curve"],alpha=0.2,color=INTER_CLR)
                ax.plot(xT,r["inter_curve"],color=INTER_CLR,lw=1.4,
                        label=f"Intermediate ({r['inter_pct']*100:.1f}%)")
            ax.fill_between(xT,r["bound_curve"],alpha=0.2,color=BOUND_CLR)
            ax.plot(xT,r["bound_curve"],color=BOUND_CLR,lw=1.4,
                    label=f"Bound ({r['bound_pct']*100:.1f}%)")
            ax.set_title(f"R² = {r['r2']:.5f}  |  RMSE = {r['rmse']:.5f}",
                         color=TEXT_DIM,fontsize=10,pad=6)
            ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Absorbance (AU)")
            ax.invert_xaxis()
            ax.legend(facecolor=SURFACE2,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
        self._draw(fn)

    def plot_residuals(self, r):
        xT=r["xT"]; res=r["yT"]-r["yfit"]; rmse=r["rmse"]
        def fn(ax):
            ax.axhline(0,color=BORDER,lw=1.0,linestyle="--")
            ax.fill_between(xT,res,alpha=0.18,color=ACCENT2)
            ax.plot(xT,res,color=ACCENT2,lw=1.3,label="Residuals")
            ax.axhline(rmse,color=TEXT_DIM,lw=0.8,linestyle=":",alpha=0.7,
                       label=f"±RMSE ({rmse:.5f})")
            ax.axhline(-rmse,color=TEXT_DIM,lw=0.8,linestyle=":",alpha=0.7)
            ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Residual (AU)")
            ax.invert_xaxis()
            ax.legend(facecolor=SURFACE2,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
        self._draw(fn)

    def placeholder(self, msg="Run the step above to see the plot"):
        def fn(ax):
            ax.text(0.5,0.5,msg,transform=ax.transAxes,
                    ha="center",va="center",color=TEXT_DIM,fontsize=12)
        self._draw(fn)

# ── Step card ─────────────────────────────────────────────────────────────────

class StepCard(QFrame):
    """A collapsible card that represents one step in the FTIR pipeline.

    Each card has three visual states:
      - Locked  (default): greyed out header, body hidden — step not yet reachable.
      - Active  (current): blue border + header, body visible — step in progress.
      - Done    (complete): green border + tick, body hidden — step completed.

    The body (controls, buttons) is hidden in locked/done states to keep the
    left panel compact.  It becomes visible only for the active step.
    """
    def __init__(self, number, title, parent=None):
        super().__init__(parent)            # QFrame base
        self.setObjectName("step_card")    # Matches the CSS rule in STYLESHEET
        self._done = False; self._active = False   # Track current visual state
        self.number = number               # Step number (1–6); shown in the circle

        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        # Header row (always visible)
        hdr = QWidget()
        hdr.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(16,12,16,12)
        self.num_lbl = QLabel(str(number))
        self.num_lbl.setFixedSize(28,28)
        self.num_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.num_lbl.setStyleSheet(
            f"background:{SURFACE2};color:{TEXT_DIM};border-radius:14px;"
            f"font-size:12px;font-weight:700;border:1px solid {BORDER};")
        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:13px;font-weight:600;")
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        hl.addWidget(self.num_lbl); hl.addWidget(self.title_lbl,1); hl.addWidget(self.status_lbl)
        outer.addWidget(hdr)

        # Body (shown when active or done)
        self.body = QWidget()
        self.body.setStyleSheet("background:transparent;")
        self.body.setVisible(False)
        bl = QVBoxLayout(self.body); bl.setContentsMargins(16,0,16,16); bl.setSpacing(10)
        self.body_layout = bl
        outer.addWidget(self.body)

    def set_number(self, n):
        """Renumber this card.  Used when a step is hidden (e.g. Step 3 in
        no-background mode) so the remaining cards stay sequentially numbered.
        Updates the visible label unless the card is in the Done state (which
        shows a tick rather than a number)."""
        self.number = n
        if not self._done:
            self.num_lbl.setText(str(n))

    def set_active(self):
        """Transition this card to the Active state (currently in progress)."""
        self._active = True; self._done = False
        # Update Qt dynamic properties so the CSS border/background rules trigger
        self.setProperty("active","true"); self.setProperty("done","false")
        self.style().unpolish(self); self.style().polish(self)   # Force CSS re-evaluation
        # Filled indigo circle for the step number
        self.num_lbl.setStyleSheet(
            f"background:{ACCENT};color:white;border-radius:14px;"
            f"font-size:12px;font-weight:700;border:none;")
        self.title_lbl.setStyleSheet(f"color:{TEXT};font-size:13px;font-weight:700;")
        self.body.setVisible(True)     # Show the controls for this step

    def set_done(self, summary=""):
        """Transition this card to the Done state (step complete)."""
        self._active = False; self._done = True
        self.setProperty("active","false"); self.setProperty("done","true")
        self.style().unpolish(self); self.style().polish(self)
        self.num_lbl.setText("✓")    # Replace number with a tick
        # Green filled circle
        self.num_lbl.setStyleSheet(
            f"background:{SUCCESS};color:#0f1117;border-radius:14px;"
            f"font-size:12px;font-weight:700;border:none;")
        self.title_lbl.setStyleSheet(f"color:{SUCCESS};font-size:13px;font-weight:600;")
        self.status_lbl.setText(summary)   # Show a brief summary in the header
        self.body.setVisible(False)        # Collapse the body to save panel space

    def set_locked(self):
        """Transition this card to the Locked state (not yet reachable)."""
        self._active = False; self._done = False
        self.setProperty("active","false"); self.setProperty("done","false")
        self.style().unpolish(self); self.style().polish(self)
        self.num_lbl.setText(str(self.number))    # Restore the step number
        # Greyed-out outlined circle
        self.num_lbl.setStyleSheet(
            f"background:{SURFACE2};color:{TEXT_DIM};border-radius:14px;"
            f"font-size:12px;font-weight:700;border:1px solid {BORDER};")
        self.title_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:13px;font-weight:600;")
        self.status_lbl.setText("")        # Clear any summary text
        self.body.setVisible(False)        # Body hidden

    def set_independent(self):
        """Independent card — always open, but visually distinct from the active pipeline step.
        Used for Step 6 (Batch Processing) which can run at any time regardless of
        where the single-sample pipeline is.  Uses teal instead of indigo so it
        doesn't look like the current pipeline step.
        """
        self._active = False; self._done = False
        self.setProperty("active","false"); self.setProperty("done","false")
        self.style().unpolish(self); self.style().polish(self)
        self.num_lbl.setText(str(self.number))
        # Teal circle — distinct from the indigo active-step circle
        self.num_lbl.setStyleSheet(
            f"background:{INTER_CLR};color:#0f1117;border-radius:14px;"
            f"font-size:12px;font-weight:700;border:none;")
        self.title_lbl.setStyleSheet(f"color:{INTER_CLR};font-size:13px;font-weight:700;")
        self.body.setVisible(True)


# ── Paste dialog ─────────────────────────────────────────────────────────────

class PasteDialog(QDialog):
    """Modal dialog to paste raw spectrum text directly."""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Paste data \u2014 {title}")
        self.setMinimumSize(540, 420)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        hint = QLabel(
            "Paste two-column data below (wavenumber, absorbance).\n"
            "Comma, tab, or space delimiters accepted. Header rows ignored automatically.\n"
            "You can paste a full JASCO .csv export \u2014 the XYDATA block is found automatically.")
        hint.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.editor = QTextEdit()
        self.editor.setPlaceholderText(
            "3800.12   0.04521\n3799.44   0.04688\n...\n\n"
            "Or paste a full JASCO export including its header block.")
        layout.addWidget(self.editor, stretch=1)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Parse & Use")
        layout.addWidget(btns)

    def get_text(self):
        return self.editor.toPlainText()


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """The main application window.

    Layout:
      Left  (fixed 436px wide, scrollable): step cards 1–6 stacked vertically.
      Right (fills remaining space):        tab widget holding plot canvases,
                                             normalization slider below plots.

    Pipeline state is stored as instance attributes (bg_raw, samp_raw, bg_proc,
    samp_proc, sub_data, trim_data, fit_result) and updated step by step.
    A step can only be run after the previous step's data exists.
    Going back to any step restores its previous result in the plot.
    """
    def __init__(self):
        super().__init__()                              # Initialise QMainWindow
        self.setWindowTitle("FTIR Bound Water Analyzer")
        self.setMinimumSize(1180, 820)                  # Minimum usable window size
        self.setStyleSheet(STYLESHEET)                  # Apply the dark theme

        # ── Pipeline state ────────────────────────────────────────────────────
        # Each attribute holds the output of the corresponding pipeline step.
        # None means that step has not been run yet.
        self.bg_raw    = None   # Step 1 output: background (xs, ys) as loaded
        self.samp_raw  = None   # Step 1 output: sample (xs, ys) as loaded
        self.bg_proc   = None   # Step 2 output: (bx, by) normalised background
        self.samp_proc = None   # Step 2 output: (sx, sy) normalised sample
        self.sub_data  = None   # Step 3 output: (sx, subtracted) background-subtracted
        self.trim_data = None   # Step 4 output: (xT, yT) trimmed + 2nd SNIP
        self.fit_result = None  # Step 5 output: full result dict from run_fit()
        self._worker   = None   # Holds the currently running StepWorker thread
        self.last_fit_params = None  # Snapshot of Step 5 params before last fit run
        self._bg_raw_orig   = None  # Original background before zero-shift correction
        self._samp_raw_orig = None  # Original sample before zero-shift correction
        self._samp_filename = "sample"  # Stem of the loaded sample file; used to tag fit results
        self._no_bg    = False  # True = sample-only mode (no background; Step 3 skipped)

        self.setAcceptDrops(True)
        self._build_ui()
        self._go_step(1)

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16,16,16,16); root.setSpacing(16)

        # Left: scrollable steps column
        steps_widget = QWidget(); steps_widget.setFixedWidth(420)
        self._sv = QVBoxLayout(steps_widget)
        self._sv.setSpacing(10); self._sv.setContentsMargins(0,0,0,0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True); scroll.setWidget(steps_widget)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(436)

        # Right: plot area with tabs
        right = QWidget()
        rv = QVBoxLayout(right); rv.setSpacing(8); rv.setContentsMargins(0,0,0,0)
        self.plot_tabs = QTabWidget()
        self.canvas_main = PlotCanvas()
        self.canvas_aux  = PlotCanvas()
        self.canvas_main.placeholder()
        self.canvas_aux.placeholder()
        self.plot_tabs.addTab(self.canvas_main, "Current step")
        self.plot_tabs.addTab(self.canvas_aux,  "Residuals")
        rv.addWidget(self.plot_tabs)

        # Norm wavenumber slider — sits under the plot x-axis
        self._norm_slider_frame = QFrame()
        self._norm_slider_frame.setStyleSheet(
            f"background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:4px;")
        nsl = QVBoxLayout(self._norm_slider_frame)
        nsl.setContentsMargins(12, 6, 12, 6); nsl.setSpacing(3)
        # Header row: label + live value
        nshl = QHBoxLayout()
        nsl_lbl = QLabel("Normalization wavenumber")
        nsl_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;font-weight:600;letter-spacing:0.5px;")
        self._norm_val_lbl = QLabel("1740 cm⁻¹")
        self._norm_val_lbl.setStyleSheet(f"color:{ACCENT2};font-size:12px;font-weight:700;")
        nshl.addWidget(nsl_lbl); nshl.addStretch(); nshl.addWidget(self._norm_val_lbl)
        nsl.addLayout(nshl)
        # Slider — range 400–4000 cm⁻¹, inverted so high wavenumber is on left
        self._norm_slider = QSlider(Qt.Orientation.Horizontal)
        self._norm_slider.setRange(400, 4000)
        self._norm_slider.setValue(1740)
        self._norm_slider.setTickInterval(200)
        self._norm_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._norm_slider.setInvertedAppearance(True)   # high wavenumber on left, matching plot
        # Tick labels row
        tick_row = QHBoxLayout(); tick_row.setContentsMargins(0,0,0,0)
        for wn in ["4000", "3500", "3000", "2500", "2000", "1500", "1000", "500"]:
            tl = QLabel(wn)
            tl.setStyleSheet(f"color:{TEXT_DIM};font-size:9px;")
            tl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tick_row.addWidget(tl)
        nsl.addWidget(self._norm_slider)
        nsl.addLayout(tick_row)
        self._norm_slider_frame.setVisible(False)   # shown only in steps 1-4
        rv.addWidget(self._norm_slider_frame)
        # Connect slider ↔ spinbox
        self._norm_slider.valueChanged.connect(self._on_norm_slider)

        root.addWidget(scroll)
        root.addWidget(right, stretch=1)

        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Load your background and sample files to begin.")

        # ── Step 1: Load files ──
        self.s1 = StepCard(1, "Load Files")
        b1 = self.s1.body_layout
        self._bg_lbl   = QLabel("No file selected")
        self._bg_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        self._samp_lbl = QLabel("No file selected")
        self._samp_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        bg_btn   = QPushButton("📂  Choose background")
        samp_btn = QPushButton("📂  Choose sample")
        bg_btn.clicked.connect(lambda: self._pick_file("bg"))
        samp_btn.clicked.connect(lambda: self._pick_file("samp"))

        self._s1_next = QPushButton("Preview Raw →"); self._s1_next.setObjectName("accent")
        self._s1_next.setDisabled(True)
        self._s1_next.clicked.connect(lambda: self._go_step(2))

        # No-background option — deconvolute a sample with no reference to subtract
        # (e.g. a pure-water spectrum). Skips Step 3 entirely.
        self._no_bg_cb = QCheckBox("No background — deconvolute sample only")
        self._no_bg_cb.setStyleSheet(f"color:{TEXT};font-size:12px;spacing:8px;")
        self._no_bg_cb.setToolTip(
            "Enable when you have no background/reference spectrum to subtract\n"
            "(e.g. deconvoluting a pure-water spectrum). Skips Step 3.")
        self._no_bg_cb.toggled.connect(self._toggle_no_bg)
        _nb_hint = QLabel("Skips background subtraction (Step 3). Use for pure-water "
                          "or already-referenced spectra.")
        _nb_hint.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;"); _nb_hint.setWordWrap(True)
        b1.addWidget(self._no_bg_cb); b1.addWidget(_nb_hint)
        b1.addWidget(hline())

        self._bg_hdr = QLabel("Background spectrum:")
        b1.addWidget(self._bg_hdr)
        _bg_btn_row = QHBoxLayout()
        _bg_paste_btn = QPushButton("\U0001f4cb  Paste")
        _bg_btn_row.addWidget(bg_btn); _bg_btn_row.addWidget(_bg_paste_btn)
        b1.addLayout(_bg_btn_row)
        b1.addWidget(self._bg_lbl)
        self._bg_meta = QLabel("")
        self._bg_meta.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        self._bg_meta.setWordWrap(True)
        b1.addWidget(self._bg_meta)
        b1.addWidget(hline())
        # Widgets that get disabled when no-background mode is on
        self._bg_widgets = [self._bg_hdr, bg_btn, _bg_paste_btn, self._bg_lbl, self._bg_meta]
        b1.addWidget(QLabel("Sample spectrum:"))
        _samp_btn_row = QHBoxLayout()
        _samp_paste_btn = QPushButton("\U0001f4cb  Paste")
        _samp_btn_row.addWidget(samp_btn); _samp_btn_row.addWidget(_samp_paste_btn)
        b1.addLayout(_samp_btn_row)
        b1.addWidget(self._samp_lbl)
        self._samp_meta = QLabel("")
        self._samp_meta.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        self._samp_meta.setWordWrap(True)
        b1.addWidget(self._samp_meta)
        _bg_paste_btn.clicked.connect(lambda: self._paste_data("bg"))
        _samp_paste_btn.clicked.connect(lambda: self._paste_data("samp"))
        b1.addWidget(hline())
        # Zero-baseline correction option
        self._zero_cb = QCheckBox("Raise minimum to zero")
        self._zero_cb.setChecked(True)
        self._zero_cb.setStyleSheet(
            f"color:{TEXT};font-size:12px;spacing:8px;")
        self._zero_cb.setToolTip(
            "If the spectrum dips below zero, shift the entire spectrum up\n"
            "so its minimum value is exactly 0. Applied on load.")
        zero_hint = QLabel(
            "Shifts spectra with negative absorbance up so the minimum is zero. "
            "Detected automatically on load.")
        zero_hint.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        zero_hint.setWordWrap(True)
        self._zero_cb.stateChanged.connect(self._reapply_zero_correction)
        b1.addWidget(self._zero_cb)
        b1.addWidget(zero_hint)
        b1.addWidget(hline())
        b1.addWidget(self._s1_next)
        self._sv.addWidget(self.s1)

        # ── Step 2: Baseline / normalization ──
        self.s2 = StepCard(2, "Baseline Correction & Normalization")
        b2 = self.s2.body_layout
        self.smooth_sb   = make_spinbox(1, 100, 20)
        self.snip_sb     = make_spinbox(10, 500, 200, step=10)
        self.norm_wav_sb = make_spinbox(400, 4000, 1740, step=10)
        self.norm_wav_sb.valueChanged.connect(
            lambda v: (self._norm_slider.blockSignals(True),
                       self._norm_slider.setValue(max(400,min(4000,v))),
                       self._norm_slider.blockSignals(False)))
        hint = QLabel("SNIP baseline → rolling mean smooth → normalize at reference wavenumber")
        hint.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;"); hint.setWordWrap(True)
        b2.addWidget(hint)
        b2.addLayout(param_row("Smoothing window (pts)", self.smooth_sb))
        b2.addLayout(param_row("SNIP iterations", self.snip_sb))
        b2.addLayout(param_row("Norm. wavenumber (cm⁻¹)", self.norm_wav_sb, "C=O → 1740"))
        for _esb in [self.smooth_sb, self.snip_sb, self.norm_wav_sb]:
            _esb.lineEdit().returnPressed.connect(self._run_step2)
        self._s2_run  = QPushButton("▶  Apply"); self._s2_run.setObjectName("accent")
        self._s2_save = QPushButton("⬇  Save CSVs"); self._s2_save.setObjectName("save")
        self._s2_save.setDisabled(True)
        self._s2_next = QPushButton("Next →"); self._s2_next.setDisabled(True)
        self._s2_back = QPushButton("← Back")
        self._s2_run.clicked.connect(self._run_step2)
        self._s2_save.clicked.connect(self._save_step2)
        self._s2_next.clicked.connect(self._go_after_step2)
        self._s2_back.clicked.connect(lambda: self._go_step(1))
        _r2a = QHBoxLayout(); _r2a.addWidget(self._s2_back); _r2a.addWidget(self._s2_run)
        _r2b = QHBoxLayout(); _r2b.addWidget(self._s2_save); _r2b.addWidget(self._s2_next)
        b2.addLayout(_r2a); b2.addLayout(_r2b)
        self._sv.addWidget(self.s2)

        # ── Step 3: Background subtraction ──
        self.s3 = StepCard(3, "Background Subtraction")
        b3 = self.s3.body_layout
        hint3 = QLabel("Subtracts the normalized background from the normalized sample.")
        hint3.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;"); hint3.setWordWrap(True)
        b3.addWidget(hint3)
        self._s3_run  = QPushButton("▶  Subtract"); self._s3_run.setObjectName("accent")
        self._s3_save = QPushButton("⬇  Save CSV"); self._s3_save.setObjectName("save")
        self._s3_save.setDisabled(True)
        self._s3_next = QPushButton("Next →"); self._s3_next.setDisabled(True)
        self._s3_back = QPushButton("← Back")
        self._s3_run.clicked.connect(self._run_step3)
        self._s3_save.clicked.connect(self._save_step3)
        self._s3_next.clicked.connect(lambda: self._go_step(4))
        self._s3_back.clicked.connect(lambda: self._go_step(2))
        _r3a = QHBoxLayout(); _r3a.addWidget(self._s3_back); _r3a.addWidget(self._s3_run)
        _r3b = QHBoxLayout(); _r3b.addWidget(self._s3_save); _r3b.addWidget(self._s3_next)
        b3.addLayout(_r3a); b3.addLayout(_r3b)
        self._sv.addWidget(self.s3)

        # ── Step 4: Trim + 2nd SNIP ──
        self.s4 = StepCard(4, "Trim O–H Region & Baseline Correct")
        b4 = self.s4.body_layout
        self.trim_s_sb = make_spinbox(2500, 3500, 3000, step=10)
        self.trim_e_sb = make_spinbox(3400, 4200, 3800, step=10)
        hint4 = QLabel("Crop to the O–H stretch region then apply a second SNIP pass to flatten the baseline.")
        hint4.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;"); hint4.setWordWrap(True)
        b4.addWidget(hint4)
        b4.addLayout(param_row("Trim start (cm⁻¹)", self.trim_s_sb))
        b4.addLayout(param_row("Trim end (cm⁻¹)",   self.trim_e_sb))
        self.trim_s_sb.valueChanged.connect(self._on_trim_changed)
        self.trim_e_sb.valueChanged.connect(self._on_trim_changed)
        self._s4_run  = QPushButton("▶  Apply"); self._s4_run.setObjectName("accent")
        self._s4_save = QPushButton("⬇  Save CSV"); self._s4_save.setObjectName("save")
        self._s4_save.setDisabled(True)
        self._s4_next = QPushButton("Next →"); self._s4_next.setDisabled(True)
        self._s4_back = QPushButton("← Back")
        self._s4_run.clicked.connect(self._run_step4)
        self._s4_save.clicked.connect(self._save_step4)
        self._s4_next.clicked.connect(lambda: self._go_step(5))
        self._s4_back.clicked.connect(lambda: self._go_step(2 if self._no_bg else 3))
        _r4a = QHBoxLayout(); _r4a.addWidget(self._s4_back); _r4a.addWidget(self._s4_run)
        _r4b = QHBoxLayout(); _r4b.addWidget(self._s4_save); _r4b.addWidget(self._s4_next)
        b4.addLayout(_r4a); b4.addLayout(_r4b)
        self._sv.addWidget(self.s4)

        # ── Step 5: Fit ──
        self.s5 = StepCard(5, "Gaussian Deconvolution")
        b5 = self.s5.body_layout

        # Fit mode
        fm_row = QHBoxLayout()
        self.triple_btn = QPushButton("Triple Gaussian"); self.triple_btn.setObjectName("toggle"); self.triple_btn.setCheckable(True); self.triple_btn.setChecked(True)
        self.double_btn = QPushButton("Double Gaussian"); self.double_btn.setObjectName("toggle"); self.double_btn.setCheckable(True)
        self.triple_btn.clicked.connect(lambda: self._set_fit_mode("triple"))
        self.double_btn.clicked.connect(lambda: self._set_fit_mode("double"))
        fm_row.addWidget(self.triple_btn); fm_row.addWidget(self.double_btn)
        b5.addLayout(fm_row)

        # Anchor mode
        an_row = QHBoxLayout()
        self.anchored_btn = QPushButton("Anchored centers"); self.anchored_btn.setObjectName("toggle"); self.anchored_btn.setCheckable(True); self.anchored_btn.setChecked(True)
        self.float_btn    = QPushButton("Floating centers");  self.float_btn.setObjectName("toggle"); self.float_btn.setCheckable(True)
        self.anchored_btn.clicked.connect(lambda: self._set_anchor("anchored"))
        self.float_btn.clicked.connect(lambda: self._set_anchor("float"))
        an_row.addWidget(self.anchored_btn); an_row.addWidget(self.float_btn)
        b5.addLayout(an_row)
        self.anchor_note = QLabel("Centers fixed — only height & sigma optimized.")
        self.anchor_note.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;"); self.anchor_note.setWordWrap(True)
        b5.addWidget(self.anchor_note)

        b5.addWidget(hline())

        # Sigma constraints
        sc_row = QHBoxLayout()
        self.constrain_btn   = QPushButton("Constrained σ");   self.constrain_btn.setObjectName("toggle"); self.constrain_btn.setCheckable(True); self.constrain_btn.setChecked(True)
        self.unconstrain_btn = QPushButton("Unconstrained σ");  self.unconstrain_btn.setObjectName("toggle"); self.unconstrain_btn.setCheckable(True)
        self.constrain_btn.clicked.connect(lambda: self._set_constrain(True))
        self.unconstrain_btn.clicked.connect(lambda: self._set_constrain(False))
        sc_row.addWidget(self.constrain_btn); sc_row.addWidget(self.unconstrain_btn)
        b5.addLayout(sc_row)
        self.sig_min_sb = make_spinbox(1, 200, 10, step=5)
        self.sig_max_sb = make_spinbox(10, 500, 200, step=10)
        b5.addLayout(param_row("Sigma min (cm⁻¹)", self.sig_min_sb, "prevents spike fits"))
        b5.addLayout(param_row("Sigma max (cm⁻¹)", self.sig_max_sb))

        b5.addWidget(hline())

        # Peak initial guesses
        b5.addWidget(QLabel("Peak initial guesses:", styleSheet=f"color:{TEXT_DIM};font-size:11px;font-weight:600;"))

        self.fc_sb = make_spinbox(3300,3700,3560,step=5)
        self.fh_sb = make_spinbox(0.001,10.0,0.1,decimals=3,step=0.01)
        self.fs_sb = make_spinbox(1,300,30)
        fl = QLabel("● Free water"); fl.setStyleSheet(f"color:{FREE_CLR};font-size:12px;font-weight:600;")
        b5.addWidget(fl)
        b5.addLayout(param_row("  Center (cm⁻¹)", self.fc_sb))
        b5.addLayout(param_row("  Height (init.)", self.fh_sb))
        b5.addLayout(param_row("  Sigma (init.)",  self.fs_sb))

        self.ic_sb  = make_spinbox(3100,3500,3375,step=5)
        self.ih_sb  = make_spinbox(0.001,10.0,0.1,decimals=3,step=0.01)
        self.isg_sb = make_spinbox(1,300,30)
        # Wrap all intermediate controls in a single container so they can be
        # hidden entirely (not just greyed) when double-Gaussian mode is selected.
        self._inter_container = QWidget()
        _icl = QVBoxLayout(self._inter_container); _icl.setContentsMargins(0,0,0,0); _icl.setSpacing(4)
        _inter_lbl = QLabel("● Intermediate water")
        _inter_lbl.setStyleSheet(f"color:{INTER_CLR};font-size:12px;font-weight:600;")
        _icl.addWidget(_inter_lbl)
        _icl.addLayout(param_row("  Center (cm⁻¹)", self.ic_sb))
        _icl.addLayout(param_row("  Height (init.)", self.ih_sb))
        _icl.addLayout(param_row("  Sigma (init.)",  self.isg_sb))
        b5.addWidget(self._inter_container)

        self.bc_sb = make_spinbox(3000,3400,3240,step=5)
        self.bh_sb = make_spinbox(0.001,10.0,1.0,decimals=3,step=0.01)
        self.bs_sb = make_spinbox(1,300,30)
        bl2 = QLabel("● Bound water"); bl2.setStyleSheet(f"color:{BOUND_CLR};font-size:12px;font-weight:600;")
        b5.addWidget(bl2)
        b5.addLayout(param_row("  Center (cm⁻¹)", self.bc_sb))
        b5.addLayout(param_row("  Height (init.)", self.bh_sb))
        b5.addLayout(param_row("  Sigma (init.)",  self.bs_sb))

        b5.addWidget(hline())
        self._s5_run   = QPushButton("▶  Fit");         self._s5_run.setObjectName("accent")
        self._s5_save  = QPushButton("⬇  Export CSV");  self._s5_save.setObjectName("save")
        self._s5_reset = QPushButton("↺  Last params"); self._s5_reset.setObjectName("save")
        self._s5_save.setDisabled(True)
        self._s5_reset.setDisabled(True)
        self._s5_reset.setToolTip("Restore the parameter values used in the previous fit")
        self._s5_back = QPushButton("← Back")
        self._s5_run.clicked.connect(self._run_step5)
        self._s5_save.clicked.connect(self._save_step5)
        self._s5_reset.clicked.connect(self._reset_fit_params)
        self._s5_back.clicked.connect(lambda: self._go_step(4))
        _r5a = QHBoxLayout(); _r5a.addWidget(self._s5_back); _r5a.addWidget(self._s5_run)
        _r5b = QHBoxLayout(); _r5b.addWidget(self._s5_save); _r5b.addWidget(self._s5_reset)
        b5.addLayout(_r5a); b5.addLayout(_r5b)

        # Results summary inside step 5
        self._res_frame = QFrame()
        self._res_frame.setStyleSheet(f"background:{SURFACE2};border:1px solid {BORDER};border-radius:8px;")
        self._res_frame.setVisible(False)
        rl = QVBoxLayout(self._res_frame); rl.setContentsMargins(12,10,12,10); rl.setSpacing(4)
        self._res_r2   = QLabel("R²: —")
        self._res_rmse = QLabel("RMSE: —")
        self._res_free = QLabel("Free: —"); self._res_free.setStyleSheet(f"color:{FREE_CLR};")
        self._res_inter= QLabel("Intermediate: —"); self._res_inter.setStyleSheet(f"color:{INTER_CLR};")
        self._res_bound= QLabel("Bound: —"); self._res_bound.setStyleSheet(f"color:{BOUND_CLR};")
        # Floating-mode fitted centers — shown only when centers were free to move
        self._res_centers_frame = QFrame()
        self._res_centers_frame.setStyleSheet(
            f"background:{SURFACE};border:1px solid {ACCENT};border-radius:6px;padding:2px;")
        self._res_centers_frame.setVisible(False)
        _cl = QVBoxLayout(self._res_centers_frame)
        _cl.setContentsMargins(10,6,10,6); _cl.setSpacing(3)
        _chead = QLabel("Fitted centers  (floating mode)")
        _chead.setStyleSheet(f"color:{ACCENT2};font-size:10px;font-weight:700;letter-spacing:0.5px;")
        self._res_c_free  = QLabel("")
        self._res_c_free.setStyleSheet(f"color:{FREE_CLR};font-size:12px;font-weight:600;")
        self._res_c_inter = QLabel("")
        self._res_c_inter.setStyleSheet(f"color:{INTER_CLR};font-size:12px;font-weight:600;")
        self._res_c_bound = QLabel("")
        self._res_c_bound.setStyleSheet(f"color:{BOUND_CLR};font-size:12px;font-weight:600;")
        for _w in [_chead, self._res_c_free, self._res_c_inter, self._res_c_bound]:
            _cl.addWidget(_w)
        # Fitted sigma (peak width) display — lets users immediately judge peak breadth
        self._res_sigmas_frame = QFrame()
        self._res_sigmas_frame.setStyleSheet(
            f"background:{SURFACE};border:1px solid {BORDER};border-radius:6px;padding:2px;")
        self._res_sigmas_frame.setVisible(False)
        _sl = QVBoxLayout(self._res_sigmas_frame)
        _sl.setContentsMargins(10,6,10,6); _sl.setSpacing(3)
        _shead = QLabel("Fitted σ  (peak width)")
        _shead.setStyleSheet(f"color:{ACCENT2};font-size:10px;font-weight:700;letter-spacing:0.5px;")
        self._res_s_free  = QLabel("")
        self._res_s_free.setStyleSheet(f"color:{FREE_CLR};font-size:12px;font-weight:600;")
        self._res_s_inter = QLabel("")
        self._res_s_inter.setStyleSheet(f"color:{INTER_CLR};font-size:12px;font-weight:600;")
        self._res_s_bound = QLabel("")
        self._res_s_bound.setStyleSheet(f"color:{BOUND_CLR};font-size:12px;font-weight:600;")
        for _w in [_shead, self._res_s_free, self._res_s_inter, self._res_s_bound]:
            _sl.addWidget(_w)

        self._res_warn = QLabel(""); self._res_warn.setStyleSheet(f"color:{DANGER};font-size:11px;"); self._res_warn.setWordWrap(True)
        self._res_copy = QPushButton("📋  Copy to clipboard")
        self._res_copy.setObjectName("save")
        self._res_copy.clicked.connect(self._copy_single_result)
        for w in [self._res_r2, self._res_rmse, self._res_free, self._res_inter,
                  self._res_bound, self._res_centers_frame, self._res_sigmas_frame,
                  self._res_warn, self._res_copy]:
            rl.addWidget(w)
        b5.addWidget(self._res_frame)
        self._sv.addWidget(self.s5)

        # ── Step 6: Batch processing ──
        self.s6 = StepCard(6, "Batch Processing")
        self.s6.set_independent()   # always open; teal distinguishes it from the active pipeline step
        b6 = self.s6.body_layout

        hint6 = QLabel(
            "Run the full pipeline on multiple sample files against one background, "
            "using the parameters set in steps 2–5. Background can be the one already "
            "loaded or a different file.")
        hint6.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        hint6.setWordWrap(True)
        b6.addWidget(hint6)

        # Background selector
        b6.addWidget(QLabel("Background file:"))
        bg6_row = QHBoxLayout()
        self._b6_bg_btn   = QPushButton("📂  Choose background")
        self._b6_use_btn  = QPushButton("↑  Use loaded bg")
        bg6_row.addWidget(self._b6_bg_btn); bg6_row.addWidget(self._b6_use_btn)
        b6.addLayout(bg6_row)
        self._b6_bg_lbl = QLabel("No background selected")
        self._b6_bg_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        b6.addWidget(self._b6_bg_lbl)

        b6.addWidget(hline())

        # Sample files list
        b6.addWidget(QLabel("Sample files:"))
        self._b6_list = QListWidget()
        self._b6_list.setFixedHeight(130)
        self._b6_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        b6.addWidget(self._b6_list)
        samp6_row = QHBoxLayout()
        self._b6_add_btn = QPushButton("➕  Add files")
        self._b6_rem_btn = QPushButton("✕  Remove")
        self._b6_rem_btn.setToolTip("Remove the selected sample file(s) from the list")
        self._b6_clr_btn = QPushButton("Clear all")
        samp6_row.addWidget(self._b6_add_btn)
        samp6_row.addWidget(self._b6_rem_btn)
        samp6_row.addWidget(self._b6_clr_btn)
        b6.addLayout(samp6_row)

        # Replicate mode — when checked, the loaded files are treated as repeat
        # measurements of the SAME sample, and the run reports mean ± SD / %RSD
        # across them instead of (or in addition to) per-file results. This is
        # the measurement reproducibility error, distinct from the fit's own
        # Monte Carlo confidence interval (which only reflects curve-fit noise).
        self._b6_replicate_cb = QCheckBox("These files are replicates of the same sample")
        self._b6_replicate_cb.setToolTip(
            "Check this if all loaded files are repeat runs of one sample.\n"
            "The summary will show mean ± SD and %RSD across replicates,\n"
            "which captures run-to-run measurement noise rather than just\n"
            "the curve-fit's own confidence interval.")
        b6.addWidget(self._b6_replicate_cb)

        b6.addWidget(hline())

        # Progress bar
        self._b6_progress = QProgressBar()
        self._b6_progress.setValue(0)
        self._b6_progress.setVisible(False)
        b6.addWidget(self._b6_progress)

        # Status list (results per file)
        self._b6_status_list = QListWidget()
        self._b6_status_list.setFixedHeight(110)
        self._b6_status_list.setVisible(False)
        b6.addWidget(self._b6_status_list)

        # Run / export row — primary "Run batch" gets its own full-width row so its
        # label isn't clipped; the two secondary actions share the row below it.
        self._b6_run  = QPushButton("▶  Run batch"); self._b6_run.setObjectName("accent")
        self._b6_save = QPushButton("⬇  Export CSV"); self._b6_save.setObjectName("save")
        self._b6_copy = QPushButton("📋  Copy table"); self._b6_copy.setObjectName("save")
        self._b6_save.setDisabled(True)
        self._b6_copy.setDisabled(True)
        b6.addWidget(self._b6_run)
        run6_row = QHBoxLayout()
        run6_row.addWidget(self._b6_save); run6_row.addWidget(self._b6_copy)
        b6.addLayout(run6_row)

        # Replicate reproducibility summary — mean ± SD / %RSD across runs,
        # shown only when "These files are replicates" was checked at run time.
        self._b6_rep_frame = QFrame()
        self._b6_rep_frame.setStyleSheet(
            f"background:{SURFACE2};border:1px solid {ACCENT};border-radius:8px;")
        self._b6_rep_frame.setVisible(False)
        _rl = QVBoxLayout(self._b6_rep_frame); _rl.setContentsMargins(12,10,12,10); _rl.setSpacing(4)
        _rhead = QLabel("Measurement error  (replicate reproducibility)")
        _rhead.setStyleSheet(f"color:{ACCENT2};font-size:10px;font-weight:700;letter-spacing:0.5px;")
        self._b6_rep_n     = QLabel("")
        self._b6_rep_n.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        self._b6_rep_free  = QLabel("");  self._b6_rep_free.setStyleSheet(f"color:{FREE_CLR};font-size:12px;font-weight:600;")
        self._b6_rep_inter = QLabel("");  self._b6_rep_inter.setStyleSheet(f"color:{INTER_CLR};font-size:12px;font-weight:600;")
        self._b6_rep_bound = QLabel("");  self._b6_rep_bound.setStyleSheet(f"color:{BOUND_CLR};font-size:12px;font-weight:600;")
        for _w in [_rhead, self._b6_rep_n, self._b6_rep_free, self._b6_rep_inter, self._b6_rep_bound]:
            _rl.addWidget(_w)
        b6.addWidget(self._b6_rep_frame)

        # Wire up
        self._b6_bg_raw   = None
        self._b6_samples  = []    # list of (name, xs, ys)
        self._b6_results  = []
        self._b6_bg_btn.clicked.connect(self._b6_pick_bg)
        self._b6_use_btn.clicked.connect(self._b6_use_loaded_bg)
        self._b6_add_btn.clicked.connect(self._b6_add_samples)
        self._b6_rem_btn.clicked.connect(self._b6_remove_selected)
        self._b6_clr_btn.clicked.connect(self._b6_clear_samples)
        self._b6_run.clicked.connect(self._b6_run_batch)
        self._b6_save.clicked.connect(self._b6_export_csv)
        self._b6_copy.clicked.connect(self._b6_copy_table)
        self._sv.addWidget(self.s6)

        self._sv.addStretch()

    # ── Norm slider + drag-and-drop ──────────────────────────────────────────────

    def _on_norm_slider(self, value):
        """Called whenever the normalization wavenumber slider moves.
        Updates the live label, syncs the spinbox (blocking its signal to prevent
        a feedback loop), and redraws the active plot with a norm-line overlay.
        """
        self._norm_val_lbl.setText(f"{value} cm⁻¹")   # Update live wavenumber label
        # Sync spinbox without triggering slider again (blockSignals breaks the loop)
        self.norm_wav_sb.blockSignals(True)
        self.norm_wav_sb.setValue(value)
        self.norm_wav_sb.blockSignals(False)
        # Redraw whichever plot is currently visible with the new norm line
        tab = self.plot_tabs.tabText(0)
        if tab in ("Raw spectra", "Raw spectrum") and self.samp_raw and (self.bg_raw or self._no_bg):
            self.canvas_main.plot_raw(self.bg_raw, self.samp_raw, norm_wav=value)
        elif tab == "Normalized spectra" and self.samp_proc:
            self.canvas_main.plot_processed(self.bg_proc, self.samp_proc, norm_wav=value)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = [u.toLocalFile() for u in event.mimeData().urls() if u.toLocalFile()]
        exts = {".txt",".csv",".dat",".asc",".jws",".dx",".spc"}
        paths = [p for p in urls if Path(p).suffix.lower() in exts]
        if not paths:
            self.status_bar.showMessage("No recognised spectrum files in the drop.")
            return
        if len(paths) == 1:
            if self.bg_raw is None:
                self._load_file_path(paths[0], "bg")
            elif self.samp_raw is None:
                self._load_file_path(paths[0], "samp")
            else:
                from PyQt6.QtWidgets import QInputDialog
                choice, ok = QInputDialog.getItem(
                    self, "Replace which file?",
                    f"{Path(paths[0]).name} — replace:",
                    ["Background", "Sample"], 0, False)
                if ok:
                    self._load_file_path(paths[0], "bg" if choice == "Background" else "samp")
        elif len(paths) >= 2:
            self._load_file_path(paths[0], "bg")
            self._load_file_path(paths[1], "samp")
            if len(paths) > 2:
                self.status_bar.showMessage(
                    f"Loaded first two files. {len(paths)-2} additional file(s) ignored.")

    def _load_file_path(self, path, key):
        raw = Path(path).read_bytes()
        try:
            if raw[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
                xs, ys, meta = parse_spectrum(raw, raw_bytes=raw)
            else:
                xs, ys, meta = parse_spectrum(raw.decode("latin-1", errors="replace"))
        except Exception as e:
            self.status_bar.showMessage(f"Could not read {Path(path).name}: {e}")
            return
        if len(xs) < 10:
            self.status_bar.showMessage(f"Too few data points in {Path(path).name}")
            return
        self._load_spectrum(key, xs, ys, meta, source=Path(path).name)
        self.status_bar.showMessage(
            f"Dropped: {Path(path).name} → {'background' if key=='bg' else 'sample'}")

    # ── No-background mode ────────────────────────────────────────────────────

    def _toggle_no_bg(self, checked):
        """Enable/disable sample-only mode.  When on, the background controls are
        greyed out, Step 3 (subtraction) is hidden, and the remaining cards are
        renumbered so they stay sequential (Step 4→3, Step 5→4)."""
        self._no_bg = checked
        for w in self._bg_widgets:
            w.setEnabled(not checked)
        # Hide the subtraction step and renumber the downstream cards
        self.s3.setVisible(not checked)
        self.s4.set_number(3 if checked else 4)
        self.s5.set_number(4 if checked else 5)
        # In sample-only mode the sample alone is enough to proceed
        if checked:
            self._s1_next.setEnabled(self.samp_raw is not None)
        else:
            self._s1_next.setEnabled(self.bg_raw is not None and self.samp_raw is not None)
        self.s1.status_lbl.setText("")
        self._go_step(1)

    def _go_after_step2(self):
        """Advance past Step 2.  In sample-only mode there is no background to
        subtract, so the normalised sample becomes the Step 3 output directly and
        we jump straight to the trim step."""
        if self._no_bg:
            self.sub_data = self.samp_proc   # pass-through: nothing to subtract
            self._go_step(4)
        else:
            self._go_step(3)

    # ── Step navigation ───────────────────────────────────────────────────────

    def _go_step(self, n):
        """Transition the UI to step n (1–5).  Updates all step card states,
        shows or hides the norm slider, and refreshes the plot to show whatever
        data is appropriate for the newly active step.  Going back to a completed
        step re-enables its Save/Next buttons so the user can save or proceed
        without re-running the step."""
        # 1 = Load, 2 = Normalise, 3 = Subtract, 4 = Trim, 5 = Fit
        cards = [None, self.s1, self.s2, self.s3, self.s4, self.s5]
        for i, c in enumerate(cards[1:], 1):
            if i < n:   c.set_done()
            elif i == n: c.set_active()
            else:        c.set_locked()
        # The no-background choice can only be changed on Step 1
        self._no_bg_cb.setEnabled(n == 1)
        # Update plot for the step being activated
        nw = self.norm_wav_sb.value()
        self._norm_slider_frame.setVisible(n <= 2)
        if n == 1:
            if self._no_bg:
                if self.samp_raw:
                    self.canvas_main.plot_raw(None, self.samp_raw, norm_wav=nw)
                    self.plot_tabs.setTabText(0, "Raw spectrum")
                    self._s1_next.setEnabled(True)
                else:
                    self.canvas_main.placeholder("Load a sample file")
            elif self.bg_raw and self.samp_raw:
                self.canvas_main.plot_raw(self.bg_raw, self.samp_raw, norm_wav=nw)
                self.plot_tabs.setTabText(0, "Raw spectra")
                self._s1_next.setEnabled(True)
            else:
                self.canvas_main.placeholder("Load background and sample files")
        elif n == 2:
            self.canvas_main.plot_raw(self.bg_raw, self.samp_raw, norm_wav=nw)
            self.plot_tabs.setTabText(0, "Raw spectrum" if self._no_bg else "Raw spectra")
            already_done = self.samp_proc is not None
            self._s2_save.setEnabled(already_done)
            self._s2_next.setEnabled(already_done)
            if already_done:
                self.canvas_main.plot_processed(self.bg_proc, self.samp_proc, norm_wav=nw)
                self.plot_tabs.setTabText(0, "Normalized spectra")
        elif n == 3:
            self.canvas_main.plot_processed(self.bg_proc, self.samp_proc)
            self.plot_tabs.setTabText(0, "Normalized spectra")
            already_done = self.sub_data is not None
            self._s3_save.setEnabled(already_done)
            self._s3_next.setEnabled(already_done)
            if already_done:
                self.canvas_main.plot_subtracted(*self.sub_data)
                self.plot_tabs.setTabText(0, "Subtracted")
        elif n == 4:
            sx,sub = self.sub_data
            self.canvas_main.plot_subtracted(sx, sub,
                trim_s=self.trim_s_sb.value(), trim_e=self.trim_e_sb.value(),
                no_bg=self._no_bg)
            self.plot_tabs.setTabText(0, "Subtracted" if not self._no_bg else "Normalized sample")
            already_done = self.trim_data is not None
            self._s4_save.setEnabled(already_done)
            self._s4_next.setEnabled(already_done)
            if already_done:
                xT,yT = self.trim_data
                self.canvas_main.plot_trimmed(xT,yT,self.trim_s_sb.value(),self.trim_e_sb.value())
                self.plot_tabs.setTabText(0, "Trimmed region")
        elif n == 5:
            xT,yT = self.trim_data
            self.canvas_main.plot_trimmed(xT,yT,self.trim_s_sb.value(),self.trim_e_sb.value())
            self.plot_tabs.setTabText(0, "Trimmed region")
            already_done = self.fit_result is not None
            self._s5_save.setEnabled(already_done)
            if already_done:
                self.canvas_main.plot_fit(self.fit_result)
                self.canvas_aux.plot_residuals(self.fit_result)
                self.plot_tabs.setTabText(0, "Fit result")
                self._res_frame.setVisible(True)

    # ── File loading ──────────────────────────────────────────────────────────

    def _reapply_zero_correction(self):
        """Called when the 'Raise minimum to zero' checkbox is toggled.
        Re-reads each spectrum from the stored original arrays and either applies
        or removes the zero shift, then updates the stored bg_raw / samp_raw.
        Immediately refreshes the raw-spectra plot so the user sees the change.
        The user still needs to re-run Step 2 for the correction to flow through
        the normalization and subtraction pipeline.
        """
        # Iterate over both background and sample in one loop
        for key, orig_attr, raw_attr, lbl, meta_lbl in [
            ("bg",   "_bg_raw_orig",   "bg_raw",   "_bg_lbl",   "_bg_meta"),
            ("samp", "_samp_raw_orig", "samp_raw", "_samp_lbl", "_samp_meta"),
        ]:
            orig = getattr(self, orig_attr)    # Retrieve the uncorrected original
            if orig is None:
                continue                        # File not yet loaded — skip
            xs, ys = orig
            ys_corr, was_shifted = self._apply_zero_shift(ys)
            if self._zero_cb.isChecked() and was_shifted:
                ys_final = ys_corr             # Checkbox ON and spectrum needed shifting
            else:
                ys_final = ys                  # Checkbox OFF or spectrum already positive
            setattr(self, raw_attr, (xs, ys_final))   # Update the live spectrum
        # Refresh the raw-spectra plot to show the change immediately
        if self.bg_raw and self.samp_raw:
            nw = self.norm_wav_sb.value()
            self.canvas_main.plot_raw(self.bg_raw, self.samp_raw, norm_wav=nw)
            self.status_bar.showMessage(
                "Zero correction " +
                ("applied" if self._zero_cb.isChecked() else "removed") +
                " — re-run Step 2 to update normalization.")

    def _paste_data(self, key):
        """Open paste dialog, parse text, load as spectrum."""
        label = "Background spectrum" if key == "bg" else "Sample spectrum"
        dlg = PasteDialog(label, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        text = dlg.get_text().strip()
        if not text:
            return
        xs, ys, meta = parse_spectrum(text)
        if len(xs) < 10:
            QMessageBox.warning(self, "Parse error",
                "Could not read enough data points from the pasted text.\n\n"
                "Expected two columns: wavenumber and absorbance.\n"
                "Comma, tab, or space delimiters are all accepted.")
            return
        self._load_spectrum(key, xs, ys, meta, source="pasted")

    def _pick_file(self, key):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select spectrum file", "",
            "Spectrum files (*.txt *.csv *.dat *.asc *.jws *.dx *.spc);;All files (*)")
        if not path: return
        raw = Path(path).read_bytes()
        try:
            if raw[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
                xs, ys, meta = parse_spectrum(raw, raw_bytes=raw)
            else:
                xs, ys, meta = parse_spectrum(raw.decode("latin-1", errors="replace"))
        except Exception as e:
            QMessageBox.critical(self, "Read error", str(e))
            return
        if len(xs) < 10:
            QMessageBox.warning(self, "Parse error",
                "Could not read enough data points.\n\n"
                "Supported: plain two-column CSV/TXT, JASCO text exports (XYDATA), "
                "and JASCO .jws binary files.")
            return
        self._load_spectrum(key, xs, ys, meta, source=Path(path).name)

    @staticmethod
    def _apply_zero_shift(ys):
        """Shift the absorbance array upward so its minimum value is exactly 0.
        Only applied if the array contains negative values (mn < 0).
        Shifting is physically valid because FTIR absorbance is a ratio — the
        absolute zero point is arbitrary relative to the background measurement.
        Returns (corrected_array, was_shifted_bool).
        """
        mn = ys.min()           # Find the most negative (or least positive) value
        if mn < 0:
            return ys - mn, True    # Shift entire array up by |mn|; min becomes 0
        return ys, False            # Already non-negative — return unchanged

    def _load_spectrum(self, key, xs, ys, meta, source=""):
        """Shared helper — store parsed spectrum and update Step 1 UI."""
        meta_parts = []
        for k in ["DATE", "TIME", "SPECTROMETER/DATA SYSTEM", "RESOLUTION",
                  "NPOINTS", "YUNITS", "FORMAT"]:
            if k in meta and str(meta[k]).strip():
                meta_parts.append(f"{k.split('/')[0].title()}: {str(meta[k]).strip()}")
        if not meta_parts:
            meta_parts = [f"{len(xs):,} pts · {xs[0]:.0f}–{xs[-1]:.0f} cm⁻¹"]
        # Apply zero correction if enabled
        ys_corr, was_shifted = self._apply_zero_shift(ys)
        if was_shifted and self._zero_cb.isChecked():
            meta_parts.append("⬆ zero-shifted")
            ys_final = ys_corr
        elif not self._zero_cb.isChecked():
            ys_final = ys
        else:
            ys_final = ys
        meta_str = "  ·  ".join(meta_parts)
        tick = f"✅ {source}" if source else "✅ Data loaded"
        if key == "bg":
            self._bg_raw_orig = (xs, ys)   # store uncorrected
            self.bg_raw = (xs, ys_final)
            self._bg_lbl.setText(tick)
            self._bg_lbl.setStyleSheet(f"color:{SUCCESS};font-size:11px;")
            self._bg_meta.setText(meta_str)
        else:
            self._samp_raw_orig = (xs, ys)
            self.samp_raw = (xs, ys_final)
            self._samp_lbl.setText(tick)
            self._samp_lbl.setStyleSheet(f"color:{SUCCESS};font-size:11px;")
            self._samp_meta.setText(meta_str)
            # Store stem (no extension) so fit results can be tagged with the filename
            self._samp_filename = Path(source).stem if source and source != "pasted" else "sample"
        if self._no_bg:
            if self.samp_raw is not None:
                self._s1_next.setEnabled(True)
                self.s1.status_lbl.setText("Sample ready — click Next")
        elif self.bg_raw is not None and self.samp_raw is not None:
            self._s1_next.setEnabled(True)
            self.s1.status_lbl.setText("Both files ready — click Next")

    # ── Step 2 ────────────────────────────────────────────────────────────────

    def _run_step2(self):
        """Trigger Step 2: SNIP baseline → smooth → normalize.
        Disables the Apply button and runs the processing on a worker thread
        to keep the UI responsive.  Results arrive via _done_step2().
        """
        self._s2_run.setEnabled(False); self._s2_run.setText("⏳ Running…")
        self.status_bar.showMessage("Applying baseline correction and normalization…")
        snip = self.snip_sb.value(); smooth = self.smooth_sb.value()
        norm  = self.norm_wav_sb.value()
        no_bg = self._no_bg
        def work():
            bg = None if no_bg else process_spectrum(*self.bg_raw, snip, smooth, norm)
            sx,sy = process_spectrum(*self.samp_raw, snip, smooth, norm)
            return bg,(sx,sy)
        self._worker = StepWorker(work)
        self._worker.finished.connect(self._done_step2)
        self._worker.error.connect(self._step_error)
        self._worker.start()

    def _done_step2(self, result):
        self.bg_proc, self.samp_proc = result
        self.canvas_main.plot_processed(self.bg_proc, self.samp_proc,
                                         norm_wav=self.norm_wav_sb.value())
        self.plot_tabs.setTabText(0, "Normalized spectra")
        self._s2_run.setEnabled(True); self._s2_run.setText("▶  Apply")
        self._s2_save.setEnabled(True); self._s2_next.setEnabled(True)
        nxt = "trim" if self._no_bg else "subtract"
        self.status_bar.showMessage(f"Step 2 complete — inspect the normalized spectrum, then save or proceed to {nxt}.")

    def _save_step2(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder to save CSVs")
        if not d: return
        sx,sy = self.samp_proc
        save_csv_two_col(Path(d)/"sample_normalized.csv", sx, sy)
        if self.bg_proc is not None:
            bx,by = self.bg_proc
            save_csv_two_col(Path(d)/"background_normalized.csv", bx, by)
        self.status_bar.showMessage(f"Saved normalized CSVs to {d}")

    # ── Step 3 ────────────────────────────────────────────────────────────────

    def _run_step3(self):
        """Trigger Step 3: background subtraction.
        Resamples the normalised background onto the sample's x-grid using linear
        interpolation (handles the case where they have slightly different grids),
        then subtracts point-by-point to give the net O-H absorbance.
        """
        self._s3_run.setEnabled(False); self._s3_run.setText("⏳ Running…")
        self.status_bar.showMessage("Subtracting background…")
        def work():
            bx,by = self.bg_proc; sx,sy = self.samp_proc
            by_r = np.interp(sx, bx, by)   # Resample bg onto sample x-grid
            return sx, sy - by_r            # Elementwise subtraction
        self._worker = StepWorker(work)
        self._worker.finished.connect(self._done_step3)
        self._worker.error.connect(self._step_error)
        self._worker.start()

    def _done_step3(self, result):
        self.sub_data = result
        sx, sub = result
        self.canvas_main.plot_subtracted(sx, sub,
            trim_s=self.trim_s_sb.value(), trim_e=self.trim_e_sb.value())
        self.plot_tabs.setTabText(0, "Subtracted")
        self._s3_run.setEnabled(True); self._s3_run.setText("▶  Subtract")
        self._s3_save.setEnabled(True); self._s3_next.setEnabled(True)
        self.status_bar.showMessage("Step 3 complete — check the subtracted spectrum, then save or proceed.")

    def _save_step3(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save subtracted spectrum", "subtracted.csv", "CSV (*.csv)")
        if not path: return
        sx, sub = self.sub_data
        save_csv_two_col(path, sx, sub, col2="delta_absorbance")
        self.status_bar.showMessage(f"Saved to {path}")

    # ── Step 4 ────────────────────────────────────────────────────────────────

    def _run_step4(self):
        """Trigger Step 4: trim to the O-H stretch region and apply a second SNIP pass.
        The trim isolates the 3000–3800 cm⁻¹ region where the three water peaks sit.
        The second SNIP corrects any remaining baseline slope within that window.
        """
        self._s4_run.setEnabled(False); self._s4_run.setText("⏳ Running…")
        self.status_bar.showMessage("Trimming and applying second SNIP correction…")
        ts = self.trim_s_sb.value(); te = self.trim_e_sb.value()
        snip = self.snip_sb.value()
        def work():
            sx, sub = self.sub_data
            ti = np.argmin(np.abs(sx - ts)); tei = np.argmin(np.abs(sx - te))
            lo, hi = sorted([ti, tei])
            xT = sx[lo:hi+1]
            yT = snip_baseline(sub[lo:hi+1], min(snip, 100))
            return xT, yT
        self._worker = StepWorker(work)
        self._worker.finished.connect(self._done_step4)
        self._worker.error.connect(self._step_error)
        self._worker.start()

    def _done_step4(self, result):
        self.trim_data = result
        xT, yT = result
        self.canvas_main.plot_trimmed(xT, yT, self.trim_s_sb.value(), self.trim_e_sb.value())
        self.plot_tabs.setTabText(0, "Trimmed region")
        self._s4_run.setEnabled(True); self._s4_run.setText("▶  Apply")
        self._s4_save.setEnabled(True); self._s4_next.setEnabled(True)
        self.status_bar.showMessage(f"Step 4 complete — {len(xT)} points in trim region. Save or proceed.")

    def _save_step4(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save trimmed spectrum", "trimmed.csv", "CSV (*.csv)")
        if not path: return
        xT, yT = self.trim_data
        save_csv_two_col(path, xT, yT)
        self.status_bar.showMessage(f"Saved to {path}")

    # ── Step 5 ────────────────────────────────────────────────────────────────

    def _run_step5(self):
        """Trigger Step 5: Gaussian deconvolution fit.
        Snapshots the current parameters for the ↺ Last params button,
        then runs run_fit() on a worker thread.  Results arrive via _done_step5().
        """
        self._s5_run.setEnabled(False); self._s5_run.setText("⏳ Fitting…")
        self._res_frame.setVisible(False)     # Hide previous results while fitting
        self.status_bar.showMessage("Running Gaussian fit…")
        xT, yT = self.trim_data
        fit_mode   = "triple" if self.triple_btn.isChecked() else "double"
        anchor     = "anchored" if self.anchored_btn.isChecked() else "float"
        constrained = self.constrain_btn.isChecked()
        params = dict(fc=self.fc_sb.value(), fh=self.fh_sb.value(), fs=self.fs_sb.value(),
                      ic=self.ic_sb.value(), ih=self.ih_sb.value(), isg=self.isg_sb.value(),
                      bc=self.bc_sb.value(), bh=self.bh_sb.value(), bs=self.bs_sb.value())
        con = dict(sig_min=self.sig_min_sb.value() if constrained else 1,
                   sig_max=self.sig_max_sb.value() if constrained else 500)
        # Snapshot current params so user can restore them after experimenting
        self.last_fit_params = dict(
            fit_mode=fit_mode, anchor=anchor, constrained=constrained,
            params=dict(params), con=dict(con))
        self._s5_reset.setEnabled(True)
        def work():
            return run_fit(xT, yT, params, fit_mode, anchor, con)
        self._worker = StepWorker(work)
        self._worker.finished.connect(self._done_step5)
        self._worker.error.connect(self._step_error)
        self._worker.start()

    def _done_step5(self, r):
        """Called on the main thread when the Gaussian fit completes.
        Updates all result labels, shows fitted centres (floating mode only),
        draws the fit overlay and residuals plots, enables the Export button,
        and displays any quality warnings in the status bar.
        """
        r.setdefault("sample_name", self._samp_filename)  # Tag with filename for clipboard/export
        self.fit_result = r           # Store for export / copy / batch comparison
        self.canvas_main.plot_fit(r)  # Overlay: data + total fit + individual Gaussians
        self.canvas_aux.plot_residuals(r)    # Residuals + ±RMSE band
        self.plot_tabs.setTabText(0, "Fit result")
        self.plot_tabs.setTabText(1, "Residuals")

        self._res_r2.setText(f"R² = {r['r2']:.5f}")
        self._res_rmse.setText(f"RMSE = {r['rmse']:.5f}")
        self._res_free.setText(f"Free:  {r['free_pct']*100:.1f}%")
        if r["fit_mode"] == "triple":
            self._res_inter.setText(f"Intermediate:  {r['inter_pct']*100:.1f}%")
            self._res_inter.setVisible(True)
        else:
            self._res_inter.setVisible(False)
        self._res_bound.setText(f"Bound:  {r['bound_pct']*100:.1f}%")
        # Show fitted centers only when floating mode was used
        c = r["centers"]
        is_floating = self.float_btn.isChecked()
        self._res_centers_frame.setVisible(is_floating)
        if is_floating:
            self._res_c_free.setText(f"  Free:          {c['free']:.2f} cm\u207b\u00b9")
            if r["fit_mode"] == "triple" and c["inter"] is not None:
                self._res_c_inter.setText(f"  Intermediate:  {c['inter']:.2f} cm\u207b\u00b9")
                self._res_c_inter.setVisible(True)
            else:
                self._res_c_inter.setVisible(False)
            self._res_c_bound.setText(f"  Bound:         {c['bound']:.2f} cm\u207b\u00b9")

        # Populate fitted \u03c3 (peak widths) \u2014 always shown after a successful fit
        fp = r["fp"]
        trip = r["fit_mode"] == "triple"
        self._res_s_free.setText(f"  Free:          \u03c3 = {abs(fp[2]):.1f} cm\u207b\u00b9")
        if trip:
            self._res_s_inter.setText(f"  Intermediate:  \u03c3 = {abs(fp[5]):.1f} cm\u207b\u00b9")
            self._res_s_inter.setVisible(True)
            self._res_s_bound.setText(f"  Bound:         \u03c3 = {abs(fp[8]):.1f} cm\u207b\u00b9")
        else:
            self._res_s_inter.setVisible(False)
            self._res_s_bound.setText(f"  Bound:         \u03c3 = {abs(fp[5]):.1f} cm\u207b\u00b9")
        self._res_sigmas_frame.setVisible(True)

        self._res_warn.setText("\n".join(r["warnings"]) if r["warnings"] else "")
        self._res_frame.setVisible(True)

        self._s5_run.setEnabled(True); self._s5_run.setText("▶  Fit")
        self._s5_save.setEnabled(True)
        if r["warnings"]:
            self.status_bar.setStyleSheet(f"color:{DANGER};")
            self.status_bar.showMessage("⚠  " + "  |  ".join(r["warnings"]))
        else:
            self.status_bar.setStyleSheet("")
            c = r["centers"]
            self.status_bar.showMessage(
                f"Fit complete — free: {c['free']:.1f}  " +
                (f"inter: {c['inter']:.1f}  " if c['inter'] else "") +
                f"bound: {c['bound']:.1f} cm⁻¹")
        self.s5.set_active()  # keep open so user can re-fit

    def _save_step5(self):
        if not self.fit_result: return
        path, _ = QFileDialog.getSaveFileName(self, "Export fit results", "ftir_fit_results.csv", "CSV (*.csv)")
        if not path: return
        r = self.fit_result
        xT,yT,yfit = r["xT"],r["yT"],r["yfit"]
        with open(path,"w",newline="") as f:
            w = csv.writer(f)
            hdr = ["wavenumber","data","total_fit","free_gaussian"]
            if r["fit_mode"]=="triple": hdr.append("intermediate_gaussian")
            hdr.append("bound_gaussian"); w.writerow(hdr)
            for i in range(len(xT)):
                row = [f"{xT[i]:.3f}",f"{yT[i]:.6f}",f"{yfit[i]:.6f}",f"{r['free_curve'][i]:.6f}"]
                if r["fit_mode"]=="triple": row.append(f"{r['inter_curve'][i]:.6f}")
                row.append(f"{r['bound_curve'][i]:.6f}"); w.writerow(row)
            w.writerow([]); w.writerow(["# Summary"])
            w.writerow(["R2",f"{r['r2']:.6f}"]); w.writerow(["RMSE",f"{r['rmse']:.6f}"])
            w.writerow(["Free area",f"{r['free_a']:.4f}"]); w.writerow(["Free %",f"{r['free_pct']*100:.2f}"])
            if r["fit_mode"]=="triple":
                w.writerow(["Inter area",f"{r['inter_a']:.4f}"]); w.writerow(["Inter %",f"{r['inter_pct']*100:.2f}"])
            w.writerow(["Bound area",f"{r['bound_a']:.4f}"]); w.writerow(["Bound %",f"{r['bound_pct']*100:.2f}"])
            fp=r["fp"]; c=r["centers"]
            w.writerow([]); w.writerow(["# Fitted parameters"])
            w.writerow(["Free center",f"{c['free']:.2f}"]); w.writerow(["Free height",f"{fp[0]:.6f}"]); w.writerow(["Free sigma",f"{abs(fp[2]):.2f}"])
            if r["fit_mode"]=="triple":
                w.writerow(["Inter center",f"{c['inter']:.2f}" if c['inter'] else "fixed"])
                w.writerow(["Inter height",f"{fp[3]:.6f}"]); w.writerow(["Inter sigma",f"{abs(fp[5]):.2f}"])
                w.writerow(["Bound center",f"{c['bound']:.2f}"]); w.writerow(["Bound height",f"{fp[6]:.6f}"]); w.writerow(["Bound sigma",f"{abs(fp[8]):.2f}"])
            else:
                w.writerow(["Bound center",f"{c['bound']:.2f}"]); w.writerow(["Bound height",f"{fp[3]:.6f}"]); w.writerow(["Bound sigma",f"{abs(fp[5]):.2f}"])
            if r["warnings"]:
                w.writerow([]); w.writerow(["# Warnings"])
                for wn in r["warnings"]: w.writerow([wn])
        self.status_bar.showMessage(f"Exported to {path}")

    # ── Fit mode / anchor / constrain toggles ─────────────────────────────────

    def _reset_fit_params(self):
        """Restore all Step 5 controls to the exact values used in the previous fit.
        Lets the user try an experimental parameter change and then snap back
        to the last known-good state with one click.
        last_fit_params is set in _run_step5() just before each fit is launched.
        """
        if not self.last_fit_params:
            return                     # No fit has been run yet — nothing to restore
        s = self.last_fit_params       # Shorthand for the saved snapshot
        p = s["params"]               # The 9 peak parameter values
        self._set_fit_mode(s["fit_mode"])       # Triple or double Gaussian
        self._set_anchor(s["anchor"])           # Anchored or floating centres
        self._set_constrain(s["constrained"])   # Sigma constraints on/off
        self.sig_min_sb.setValue(s["con"]["sig_min"])   # Sigma lower bound
        self.sig_max_sb.setValue(s["con"]["sig_max"])   # Sigma upper bound
        # Restore the nine initial-guess spinboxes
        self.fc_sb.setValue(p["fc"]); self.fh_sb.setValue(p["fh"]); self.fs_sb.setValue(p["fs"])
        self.ic_sb.setValue(p["ic"]); self.ih_sb.setValue(p["ih"]); self.isg_sb.setValue(p["isg"])
        self.bc_sb.setValue(p["bc"]); self.bh_sb.setValue(p["bh"]); self.bs_sb.setValue(p["bs"])
        self.status_bar.showMessage("Parameters restored to previous fit values.")

    def _set_fit_mode(self, mode):
        self.triple_btn.setChecked(mode=="triple"); self.double_btn.setChecked(mode=="double")
        # Hide the entire intermediate container in double-Gaussian mode so the panel
        # stays clean; just disabling/greying left phantom space and confused users.
        self._inter_container.setVisible(mode == "triple")

    def _set_anchor(self, mode):
        self.anchored_btn.setChecked(mode=="anchored"); self.float_btn.setChecked(mode=="float")
        self.anchor_note.setText("Centers fixed — only height & sigma optimized."
                                  if mode=="anchored"
                                  else "Centers used as initial guesses, allowed to shift.")

    def _set_constrain(self, on):
        self.constrain_btn.setChecked(on); self.unconstrain_btn.setChecked(not on)
        self.sig_min_sb.setEnabled(on); self.sig_max_sb.setEnabled(on)

    # ── Batch handlers ───────────────────────────────────────────────────────────

    def _b6_pick_bg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select background file", "",
            "Spectrum files (*.txt *.csv *.dat *.asc *.jws *.dx *.spc);;All files (*)")
        if not path: return
        raw = Path(path).read_bytes()
        if raw[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
            xs, ys, _ = parse_spectrum(raw, raw_bytes=raw)
        else:
            xs, ys, _ = parse_spectrum(raw.decode("latin-1", errors="replace"))
        if len(xs) < 10:
            QMessageBox.warning(self, "Parse error", "Could not read background file.")
            return
        self._b6_bg_raw = (xs, ys)
        self._b6_bg_lbl.setText(f"✅ {Path(path).name}")
        self._b6_bg_lbl.setStyleSheet(f"color:{SUCCESS};font-size:11px;")

    def _b6_use_loaded_bg(self):
        if self.bg_raw is None:
            QMessageBox.warning(self, "No background", "Load a background in Step 1 first.")
            return
        self._b6_bg_raw = self.bg_raw
        self._b6_bg_lbl.setText("✅ Using background from Step 1")
        self._b6_bg_lbl.setStyleSheet(f"color:{SUCCESS};font-size:11px;")

    def _b6_add_samples(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select sample files", "",
            "Spectrum files (*.txt *.csv *.dat *.asc *.jws *.dx *.spc);;All files (*)")
        for path in paths:
            name = Path(path).name
            # Skip duplicates
            if any(n == name for n, *_ in self._b6_samples):
                continue
            try:
                raw = Path(path).read_bytes()
                if raw[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
                    xs, ys, _ = parse_spectrum(raw, raw_bytes=raw)
                else:
                    xs, ys, _ = parse_spectrum(raw.decode("latin-1", errors="replace"))
                if len(xs) < 10:
                    raise ValueError("Too few data points")
                self._b6_samples.append((name, xs, ys))
                item = QListWidgetItem(f"  {name}")
                item.setForeground(QColor(TEXT))
                self._b6_list.addItem(item)
            except Exception as e:
                item = QListWidgetItem(f"  ⚠ {name}  ({e})")
                item.setForeground(QColor(DANGER))
                self._b6_list.addItem(item)

    def _b6_remove_selected(self):
        rows = sorted([self._b6_list.row(i) for i in self._b6_list.selectedItems()], reverse=True)
        for r in rows:
            self._b6_list.takeItem(r)
            if r < len(self._b6_samples):
                self._b6_samples.pop(r)

    def _b6_clear_samples(self):
        self._b6_list.clear()
        self._b6_samples.clear()

    def _b6_collect_settings(self):
        fit_mode    = "triple" if self.triple_btn.isChecked() else "double"
        anchor      = "anchored" if self.anchored_btn.isChecked() else "float"
        constrained = self.constrain_btn.isChecked()
        return {
            "snip":     self.snip_sb.value(),
            "smooth":   self.smooth_sb.value(),
            "norm_wav": self.norm_wav_sb.value(),
            "trim_s":   self.trim_s_sb.value(),
            "trim_e":   self.trim_e_sb.value(),
            "fit_mode": fit_mode,
            "anchor_mode": anchor,
            "constraints": {
                "sig_min": self.sig_min_sb.value() if constrained else 1,
                "sig_max": self.sig_max_sb.value() if constrained else 500,
            },
            "params": dict(
                fc=self.fc_sb.value(), fh=self.fh_sb.value(), fs=self.fs_sb.value(),
                ic=self.ic_sb.value(), ih=self.ih_sb.value(), isg=self.isg_sb.value(),
                bc=self.bc_sb.value(), bh=self.bh_sb.value(), bs=self.bs_sb.value(),
            )
        }

    def _b6_run_batch(self):
        if self._b6_bg_raw is None:
            QMessageBox.warning(self, "No background",
                "Choose a background file or click '↑ Use loaded bg'.")
            return
        if not self._b6_samples:
            QMessageBox.warning(self, "No samples", "Add at least one sample file.")
            return

        self._b6_run.setEnabled(False)
        self._b6_run.setText("⏳  Running…")
        self._b6_save.setDisabled(True)
        self._b6_results = []
        self._b6_status_list.clear()
        self._b6_status_list.setVisible(True)
        self._b6_progress.setMaximum(len(self._b6_samples))
        self._b6_progress.setValue(0)
        self._b6_progress.setVisible(True)
        self.status_bar.showMessage(f"Batch: processing 0 / {len(self._b6_samples)}…")

        # Mark each list item as pending
        for i in range(self._b6_list.count()):
            item = self._b6_list.item(i)
            if item:
                name = item.text().strip()
                item.setText(f"  ⏳ {name}")

        settings = self._b6_collect_settings()
        self._batch_worker = BatchWorker(self._b6_bg_raw, self._b6_samples, settings)
        self._batch_worker.progress.connect(self._b6_on_progress)
        self._batch_worker.finished.connect(self._b6_on_finished)
        self._batch_worker.start()

    def _b6_on_progress(self, idx, name, result):
        self._b6_progress.setValue(idx + 1)
        n_total = len(self._b6_samples)
        self.status_bar.showMessage(f"Batch: {idx+1} / {n_total} — {name}")

        # Update the file list item
        if idx < self._b6_list.count():
            list_item = self._b6_list.item(idx)
            if isinstance(result, dict) and "error" not in result:
                fm = result.get("fit_mode","triple")
                pcts = (f"free {result['free_pct']*100:.1f}%  "
                        + (f"inter {result['inter_pct']*100:.1f}%  " if fm=="triple" else "")
                        + f"bound {result['bound_pct']*100:.1f}%  "
                        + f"R²={result['r2']:.4f}")
                warn = "  ⚠" if result.get("warnings") else ""
                list_item.setText(f"  ✅ {name}  —  {pcts}{warn}")
                list_item.setForeground(QColor(SUCCESS))
            else:
                err = result if isinstance(result, str) else result.get("error","?")
                list_item.setText(f"  ✗ {name}  —  {err}")
                list_item.setForeground(QColor(DANGER))

        # Add to status summary list
        if isinstance(result, dict) and "error" not in result:
            fm = result.get("fit_mode","triple")
            summary = (f"{name}  |  "
                       f"free {result['free_pct']*100:.1f}%  "
                       + (f"inter {result['inter_pct']*100:.1f}%  " if fm=="triple" else "")
                       + f"bound {result['bound_pct']*100:.1f}%  "
                       f"R²={result['r2']:.4f}  RMSE={result['rmse']:.5f}")
            si = QListWidgetItem(summary)
            si.setForeground(QColor(TEXT))
            self._b6_status_list.addItem(si)
        else:
            err = result if isinstance(result, str) else result.get("error","?")
            si = QListWidgetItem(f"✗ {name}  —  {err}")
            si.setForeground(QColor(DANGER))
            self._b6_status_list.addItem(si)

        self._b6_results.append(result)

    def _b6_on_finished(self, results):
        self._b6_run.setEnabled(True)
        self._b6_run.setText("▶  Run batch")
        n_ok  = sum(1 for r in results if isinstance(r,dict) and "error" not in r)
        n_err = len(results) - n_ok
        self.status_bar.setStyleSheet("" if n_err == 0 else f"color:{DANGER};")
        self.status_bar.showMessage(
            f"Batch complete — {n_ok} succeeded, {n_err} failed.")
        if n_ok > 0:
            self._b6_save.setEnabled(True)
            self._b6_copy.setEnabled(True)
        if self._b6_replicate_cb.isChecked():
            self._show_replicate_stats(results)
        else:
            self._b6_rep_frame.setVisible(False)

    @staticmethod
    def _replicate_stats(results):
        """Mean / sample SD / %RSD across replicate runs of one sample, per water
        fraction. Returns None if fewer than 2 successful runs (SD undefined).
        This is the measurement's run-to-run reproducibility error — distinct
        from each individual fit's own Monte Carlo confidence interval, which
        only reflects curve-fit noise within a single spectrum.
        """
        ok = [r for r in results if isinstance(r, dict) and "error" not in r]
        if len(ok) < 2:
            return None
        trip = ok[0]["fit_mode"] == "triple"
        def stats(key):
            vals = np.array([r[key] for r in ok]) * 100   # fractions -> percent
            mean = vals.mean(); sd = vals.std(ddof=1)      # sample SD (n-1)
            rsd = (sd / mean * 100) if mean else float("nan")
            return mean, sd, rsd
        out = {"n": len(ok), "free": stats("free_pct"), "bound": stats("bound_pct")}
        out["inter"] = stats("inter_pct") if trip else None
        out["fit_mode"] = ok[0]["fit_mode"]
        return out

    def _show_replicate_stats(self, results):
        stats = self._replicate_stats(results)
        if stats is None:
            self._b6_rep_frame.setVisible(False)
            self.status_bar.showMessage(
                self.status_bar.currentMessage() +
                "  (need ≥2 successful runs to compute replicate error)")
            return
        self._b6_rep_n.setText(f"n = {stats['n']} replicate runs")
        fm, sf, rf = stats["free"]
        self._b6_rep_free.setText(f"Free:  {fm:.1f}% ± {sf:.1f}  (RSD {rf:.1f}%)")
        if stats["inter"] is not None:
            im, si, ri = stats["inter"]
            self._b6_rep_inter.setText(f"Intermediate:  {im:.1f}% ± {si:.1f}  (RSD {ri:.1f}%)")
            self._b6_rep_inter.setVisible(True)
        else:
            self._b6_rep_inter.setVisible(False)
        bm, sb, rb = stats["bound"]
        self._b6_rep_bound.setText(f"Bound:  {bm:.1f}% ± {sb:.1f}  (RSD {rb:.1f}%)")
        self._b6_rep_frame.setVisible(True)

    def _b6_export_csv(self):
        if not self._b6_results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save batch summary", "batch_results.csv", "CSV (*.csv)")
        if not path:
            return

        ok = [r for r in self._b6_results if isinstance(r,dict) and "error" not in r]
        if not ok:
            return

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            fm = ok[0]["fit_mode"]
            # Header row
            hdr = ["sample", "r2", "rmse",
                   "free_pct",
                   "free_area",
                   "inter_pct",
                   "inter_area",
                   "bound_pct",
                   "bound_area",
                   "free_center", "free_sigma",
                   "inter_center", "inter_sigma",
                   "bound_center", "bound_sigma",
                   "warnings"]
            w.writerow(hdr)
            for r in ok:
                fp  = r["fp"]
                c   = r["centers"]
                is_trip = r["fit_mode"] == "triple"
                row = [
                    r["sample_name"],
                    f"{r['r2']:.6f}", f"{r['rmse']:.6f}",
                    f"{r['free_pct']*100:.3f}",
                    f"{r['free_a']:.4f}",
                    f"{r['inter_pct']*100:.3f}" if is_trip else "n/a",
                    f"{r['inter_a']:.4f}" if is_trip else "n/a",
                    f"{r['bound_pct']*100:.3f}",
                    f"{r['bound_a']:.4f}",
                    f"{c['free']:.2f}",  f"{abs(fp[2]):.2f}",
                    f"{c['inter']:.2f}" if (is_trip and c['inter']) else "fixed",
                    f"{abs(fp[5]):.2f}" if is_trip else "n/a",
                    f"{c['bound']:.2f}", f"{abs(fp[8] if is_trip else fp[5]):.2f}",
                    "; ".join(r.get("warnings", [])) or "",
                ]
                w.writerow(row)

            # Replicate reproducibility summary, if the user flagged these as
            # repeat runs of one sample (mean ± SD / %RSD = measurement error)
            if self._b6_replicate_cb.isChecked():
                stats = self._replicate_stats(self._b6_results)
                if stats is not None:
                    w.writerow([])
                    w.writerow(["# Measurement error (replicate reproducibility)"])
                    w.writerow(["n_replicates", stats["n"]])
                    fm_, sf_, rf_ = stats["free"]
                    w.writerow(["free_pct_mean", f"{fm_:.3f}"]); w.writerow(["free_pct_sd", f"{sf_:.3f}"]); w.writerow(["free_pct_rsd", f"{rf_:.2f}"])
                    if stats["inter"] is not None:
                        im_, si_, ri_ = stats["inter"]
                        w.writerow(["inter_pct_mean", f"{im_:.3f}"]); w.writerow(["inter_pct_sd", f"{si_:.3f}"]); w.writerow(["inter_pct_rsd", f"{ri_:.2f}"])
                    bm_, sb_, rb_ = stats["bound"]
                    w.writerow(["bound_pct_mean", f"{bm_:.3f}"]); w.writerow(["bound_pct_sd", f"{sb_:.3f}"]); w.writerow(["bound_pct_rsd", f"{rb_:.2f}"])

            # Also write errors if any
            errs = [r for r in self._b6_results if not (isinstance(r,dict) and "error" not in r)]
            if errs:
                w.writerow([])
                w.writerow(["# Failed samples"])
                for r in errs:
                    if isinstance(r, dict):
                        w.writerow([r.get("sample_name","?"), r.get("error","?")])

        self.status_bar.showMessage(f"Batch CSV exported to {path}")

    # ── Copy to clipboard ────────────────────────────────────────────────────────

    @staticmethod
    def _result_to_tsv_row(r, include_header=False):
        """Format a single run_fit() result dict as a tab-separated row suitable
        for pasting directly into Excel.  Returns (header_str, data_str).
        Tab separation means Excel splits each value into its own column automatically.
        The header row should only be included once (for the first pasted result);
        subsequent results can be pasted below it.
        """
        # Note: include_header param is declared for API clarity but the method
        # always returns both strings — the caller decides whether to include the header.
        fp   = r["fp"]
        c    = r["centers"]
        trip = r["fit_mode"] == "triple"

        header = "	".join([
            "Sample",
            "R2", "RMSE",
            "Free %", "Free area",  "Free center", "Free sigma",
            "Inter %", "Inter area", "Inter center","Inter sigma",
            "Bound %", "Bound area",  "Bound center","Bound sigma",
            "Warnings"
        ])
        data = "	".join([
            r.get("sample_name",""),
            f"{r['r2']:.5f}", f"{r['rmse']:.5f}",
            f"{r['free_pct']*100:.3f}",  f"{r['free_a']:.4f}",
            f"{c['free']:.2f}", f"{abs(fp[2]):.2f}",
            f"{r['inter_pct']*100:.3f}" if trip else "",
            f"{r['inter_a']:.4f}" if trip else "",
            f"{c['inter']:.2f}" if (trip and c["inter"]) else "fixed",
            f"{abs(fp[5]):.2f}" if trip else "",
            f"{r['bound_pct']*100:.3f}", f"{r['bound_a']:.4f}",
            f"{c['bound']:.2f}",
            f"{abs(fp[8] if trip else fp[5]):.2f}",
            "; ".join(r.get("warnings",[]))
        ])
        return header, data

    def _copy_single_result(self):
        """Copy the current Step 5 fit result to the system clipboard as a
        tab-separated table (header row + data row).  Paste directly into Excel
        with Ctrl+V — each column lands in its own cell automatically.
        For building up a comparison table, paste the header row once, then
        paste only the data row for subsequent samples.
        """
        if not self.fit_result:
            return
        r = self.fit_result
        header, data = self._result_to_tsv_row(r) # Build the 19-column TSV
        text = header + "\n" + data              # Combine header and data with newline
        QApplication.clipboard().setText(text)    # Put on system clipboard
        self.status_bar.showMessage(
            "Copied to clipboard — paste into Excel (Ctrl+V). "
            "First row is headers, second row is values.")

    def _b6_copy_table(self):
        """Copy all successful batch results to the clipboard as a TSV table
        with one header row followed by one data row per sample.
        Paste into Excel to immediately get a formatted summary table.
        Failed samples (those with 'error' keys) are excluded from the copy.
        """
        ok = [r for r in self._b6_results
              if isinstance(r, dict) and "error" not in r]  # Filter out failures
        if not ok:
            return
        rows = []
        header, _ = self._result_to_tsv_row(ok[0])  # Build header from first result
        rows.append(header)                           # One header row at the top
        for r in ok:
            _, data = self._result_to_tsv_row(r)     # One data row per sample
            rows.append(data)
        QApplication.clipboard().setText("\n".join(rows))  # Newline-separated rows
        self.status_bar.showMessage(
            f"Copied {len(ok)} rows to clipboard — paste into Excel (Ctrl+V).")

    def _on_trim_changed(self):
        """Called whenever the trim-start or trim-end spinbox changes.
        Immediately redraws the subtracted spectrum with a shaded band showing the
        current trim region, so the user can see where the fit will be applied
        before clicking Apply.  Only runs if Step 3 data exists and the subtracted
        spectrum is currently visible.
        """
        if self.sub_data is None:
            return   # Step 3 not yet run — nothing to update
        tab = self.plot_tabs.tabText(0)
        if tab in ("Subtracted", "Normalized sample", "Trimmed region"):
            sx, sub = self.sub_data
            # Redraw with the shaded trim region overlay
            self.canvas_main.plot_subtracted(
                sx, sub,
                trim_s=self.trim_s_sb.value(),
                trim_e=self.trim_e_sb.value(),
                no_bg=self._no_bg)
            self.plot_tabs.setTabText(0, "Normalized sample" if self._no_bg else "Subtracted")

    # ── Error handler ─────────────────────────────────────────────────────────

    def _step_error(self, msg):
        """Called by StepWorker.error signal when any step fails.
        Re-enables all run buttons (so the user can try again with different
        parameters) and shows the Python exception message in a dialog.
        """
        for btn in [self._s2_run, self._s3_run, self._s4_run, self._s5_run]:
            btn.setEnabled(True)             # Re-enable all run buttons
        for btn in [self._s2_run, self._s3_run, self._s4_run]:
            btn.setText("▶  Apply")          # Restore button labels
        self._s5_run.setText("▶  Fit")
        QMessageBox.critical(self, "Error", msg)   # Show the exception to the user
        self.status_bar.showMessage("Step failed — see error dialog.")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Entry point.  Creates the Qt application, sets the Fusion base style
    (a neutral cross-platform style that our dark STYLESHEET is applied on top of),
    builds the main window, and starts the Qt event loop.
    sys.exit ensures the process returns Qt's exit code to the shell.
    """
    app = QApplication(sys.argv)   # One QApplication per process
    app.setStyle("Fusion")          # Neutral base style; STYLESHEET overrides appearance
    win = MainWindow()             # Build the full UI
    win.show()                     # Make the window visible
    sys.exit(app.exec())           # Enter the Qt event loop; blocks until window closes

if __name__ == "__main__":
    main()   # Only run when executed directly, not when imported as a module
