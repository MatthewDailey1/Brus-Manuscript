# Run this script after running the Viewer for Integration to 1D.py script and getting isolated 1D curves.
# This will allow you to select a specific frame, perform background subtraction, and then isolate and fit peaks for further analysis.
# You will run this script in "ITO Calibration Mode" to refine your calibration parameters based on the ITO reference peaks first
# then select a sample from your dataset to analyze using the new calibration curves from the terminal output

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as messagebox

import matplotlib
matplotlib.use('TkAgg')
plt.style.use('seaborn-v0_8-whitegrid')

# =========================================================================
# X-RAY CONVERSION CONSTANTS (ALS Beamline 12.3: 10.0 keV energy)
# =========================================================================
ENERGY_KEV = 10.0
WAVELENGTH_A = 12.3984 / ENERGY_KEV  # λ ≈ 1.23984 Å
K_SHAPE_FACTOR = 0.90                 # Standard cubic crystal domain factor

# =========================================================================
# CALIBRATION PARAMETERS - Paste your updated terminal output blocks here
# =========================================================================
DEFAULT_ITO_CENTERS = [1.45074, 2.06444, 2.37281, 3.37706, 3.97509]
DEFAULT_ITO_FWHMS   = [0.02457, 0.05306, 0.03215, 0.04098, 0.03963]
DEFAULT_ITO_ETAS    = [0.00000, 0.00000, 0.00000, 0.00000, 0.06062]

GLOBAL_CALIBRATION_FILE = "ito_calibration.npz"
CALIBRATION_DATA = {
    'centers': DEFAULT_ITO_CENTERS,
    'fwhms': DEFAULT_ITO_FWHMS,
    'etas': DEFAULT_ITO_ETAS
}

SAVE_FIGURES = False
SAVE_PEAK_DATA = False

def load_calibration():
    """Load calibration data from file or use defaults"""
    try:
        if os.path.exists(GLOBAL_CALIBRATION_FILE):
            data = np.load(GLOBAL_CALIBRATION_FILE)
            return {
                'centers': data['centers'].tolist(),
                'fwhms': data['fwhms'].tolist(),
                'etas': data['etas'].tolist()
            }
    except:
        pass
    return CALIBRATION_DATA

def save_calibration(centers, fwhms, etas):
    """Save calibration data to file"""
    np.savez(GLOBAL_CALIBRATION_FILE, centers=centers, fwhms=fwhms, etas=etas)
    print(f"\n[SAVED] Calibration data to {GLOBAL_CALIBRATION_FILE}")

# Load active calibration parameters
ITO_CENTERS, ITO_FWHMS, ITO_ETAS = load_calibration().values()

def twotheta_to_q(twotheta_deg, wavelength):
    """Convert 2-Theta (degrees) to Q-space (Å^-1)"""
    twotheta_rad = np.radians(twotheta_deg)
    return (4.0 * np.pi / wavelength) * np.sin(twotheta_rad / 2.0)

def q_to_twotheta_rad(q, wavelength):
    """Convert Q-space (Å^-1) back to 2-Theta (radians)"""
    return 2.0 * np.arcsin((q * wavelength) / (4.0 * np.pi))

# =========================================================================
# INTERACTIVE USER ROUTING: CALIBRATION VS SAMPLE MODE
# =========================================================================
def ask_operation_mode():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    response = messagebox.askyesno(
        "Select Operation Mode",
        "Is this dataset for ITO Calibration?\n\n"
        "Click 'Yes' to calibrate using ITO peaks over the wide range.\n"
        "Click 'No' to analyze isolated sample data using synchrotron-optimized curves."
    )
    root.destroy()
    return response

IS_ITO_CALIBRATION = ask_operation_mode()

# DYNAMIC WINDOW DEFINITIONS: Calibrate wide, analyze isolated (100) peak narrow
if IS_ITO_CALIBRATION:
    print("\n[MODE] -> ITO CALIBRATION SCAN MODE ACTIVE (Wide Range view open)")
    Q_MIN = 0.7
    Q_MAX = 4.5
    TARGET_PEAKS = {
        'ITO-1': {'q_range': (1.4, 1.6)},
        'ITO-2': {'q_range': (2.0, 2.2)},
        'ITO-3': {'q_range': (2.3, 2.6)},
        'ITO-4': {'q_range': (3.3, 3.6)},
        'ITO-5': {'q_range': (3.9, 4.3)}
    }
else:
    print("\n[MODE] -> SAMPLE ANALYSIS ACTIVE (Isolated 100 Peak Tracking)")
    print(f"Using baseline calibration with {len(ITO_CENTERS)} ITO reference coordinates")
    Q_MIN = 0.90
    Q_MAX = 1.15
    TARGET_PEAKS = {
        '100': {'q_range': (0.9, 1.15), 'sq_sum': 1}
    }

    # --- FIX A: POLYNOMIAL CLAMPING ---
    if len(ITO_CENTERS) >= 3:
        raw_fwhm_poly = np.poly1d(np.polyfit(ITO_CENTERS, ITO_FWHMS, deg=2))
        raw_eta_poly = np.poly1d(np.polyfit(ITO_CENTERS, ITO_ETAS, deg=2))
    else:
        raw_fwhm_poly = np.poly1d(np.polyfit(ITO_CENTERS, ITO_FWHMS, deg=1))
        raw_eta_poly = np.poly1d(np.polyfit(ITO_CENTERS, ITO_ETAS, deg=1))
        print("[WARNING] Bare minimum calibration points available, using linear interpolation curves")

    min_q_cal = min(ITO_CENTERS) if len(ITO_CENTERS) > 0 else 1.0

    def fwhm_irf_curve(q):
        return raw_fwhm_poly(max(q, min_q_cal))

    def eta_irf_curve(q):
        return raw_eta_poly(max(q, min_q_cal))

SNIP_ITERATIONS = 12
PEAK_PROMINENCE_FACTOR = 0.03
PEAK_DISTANCE = 12

def select_file_via_popup(title_text, file_types=[("All Files", "*.*")]):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(title=title_text, filetypes=file_types)
    root.destroy()
    return Path(file_path) if file_path else None

def calculate_snip_background(y_data, iterations=12):
    bg = np.copy(y_data)
    n = len(y_data)

    if np.any(np.isnan(bg)):
        x = np.arange(n)
        mask = ~np.isnan(bg)
        if np.sum(mask) > 1:
            bg = np.interp(x, x[mask], bg[mask])
        else:
            bg = np.zeros_like(bg)

    for p in range(1, iterations + 1):
        if 2*p >= n:
            break
        left_shift = bg[0 : n - 2*p]
        right_shift = bg[2*p : n]
        average_neighbor = (left_shift + right_shift) / 2.0
        center_region = bg[p : n - p]
        bg[p : n - p] = np.where(center_region > average_neighbor, average_neighbor, center_region)

    for p in range(iterations - 1, 0, -1):
        if 2*p >= n:
            continue
        left_shift = bg[0 : n - 2*p]
        right_shift = bg[2*p : n]
        average_neighbor = (left_shift + right_shift) / 2.0
        center_region = bg[p : n - p]
        bg[p : n - p] = np.where(center_region > average_neighbor, average_neighbor, center_region)

    return bg

def pseudo_voigt(x, amp, cen, fwhm, eta=0.5):
    if fwhm <= 0:
        fwhm = 0.001

    sigma = fwhm / 2.35482
    g_profile = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((x - cen)**2) / (2.0 * sigma**2))
    max_g = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    g_profile /= max_g

    gamma = fwhm / 2.0
    l_profile = (1.0 / np.pi) * (gamma / ((x - cen)**2 + gamma**2))
    max_l = 1.0 / (np.pi * gamma)
    l_profile /= max_l

    return amp * ((eta * l_profile) + ((1.0 - eta) * g_profile))

def multi_pv_model(x, *params):
    c0 = params[-2]
    c1 = params[-1]
    y = c0 + c1 * x

    for i in range(0, len(params) - 2, 4):
        y += pseudo_voigt(x, params[i], params[i+1], params[i+2], params[i+3])
    return y

def detect_peaks_robust(y_data, x_data, prominence_factor=0.03, min_distance=12):
    if len(y_data) == 0:
        return []
    y_sorted = np.sort(y_data)
    median = np.median(y_sorted)
    mad = np.median(np.abs(y_sorted - median))
    threshold = max(median + 2.5 * mad, np.max(y_data) * prominence_factor)

    peaks, _ = find_peaks(
        y_data,
        print("Detecting peaks..."),
        prominence=threshold,
        distance=min_distance,
        width=3,
        rel_height=0.5
    )
    edge_margin = 5
    peaks = peaks[(peaks > edge_margin) & (peaks < len(y_data) - edge_margin)]
    return peaks

# =========================================================================
# FILE INGESTION
# =========================================================================
print("Select your compiled 1D integrated dataset (.csv)...")
cache_csv_path = select_file_via_popup("Select Integrated 1D Dataset CSV", [("CSV Files", "*.csv")])
if not cache_csv_path:
    sys.exit("No data file selected.")

folder_path = cache_csv_path.parent
run_structural_hkl = False
a_mapi = None

if not IS_ITO_CALIBRATION:
    print("\nSelect your Pure Cubic MAPI CIF file (Click Cancel to SKIP)...")
    mapi_cif = select_file_via_popup("Select Pure Cubic MAPI CIF (Cancel to Skip)", [("CIF Files", "*.cif")])
    if mapi_cif:
        try:
            from diffpy.structure import loadStructure
            a_mapi = loadStructure(str(mapi_cif)).lattice.a
            run_structural_hkl = True
            print(f"[INFO] MAPI lattice parameter a = {a_mapi:.4f} Å")
        except Exception as e:
            print(f"[WARNING] Structural analysis disabled: {e}")

print(f"\n[INFO] Loading raw matrix data from {cache_csv_path.name}...")
raw_df = pd.read_csv(cache_csv_path)

time_headers = list(raw_df.columns[1:])
frame_metadata = list(raw_df.iloc[0, 1:].values)
frame_names = [f"{f} ({t})" for f, t in zip(frame_metadata, time_headers)]

cleaned_df = raw_df.iloc[1:].copy()
raw_twotheta = pd.to_numeric(cleaned_df.iloc[:, 0]).values

cut_mask = raw_twotheta >= 5.0
x_vals_pre = twotheta_to_q(raw_twotheta[cut_mask], WAVELENGTH_A)

q_clip_mask = (x_vals_pre >= Q_MIN) & (x_vals_pre <= Q_MAX)
x_vals = x_vals_pre[q_clip_mask]
axis_label = r"Scattering Vector q ($\AA^{-1}$)"

raw_matrix_pre = cleaned_df.iloc[:, 1:].apply(pd.to_numeric).values
raw_matrix_pre = raw_matrix_pre[cut_mask, :]
raw_matrix = raw_matrix_pre[q_clip_mask, :]
total_frames = raw_matrix.shape[1]

# =========================================================================
# SUBSTRATE REFERENCE SELECTION (PASSIVE OVERLAY)
# =========================================================================
has_passive_substrate = False
substrate_vector = None

if not IS_ITO_CALIBRATION:
    print("\n[OPTIONAL] Select a bare ITO frame to view as a background reference overlay...")
    fig_sub, ax_sub = plt.subplots(figsize=(10, 4))
    plt.subplots_adjust(bottom=0.25)
    line_sub, = ax_sub.plot(x_vals, raw_matrix[:, 0], '-', color='crimson', lw=1.5)
    ax_sub.set_xlim(x_vals.min(), x_vals.max())
    ax_sub.set_ylabel('Intensity [a.u.]')
    fig_sub.suptitle(f"Select Substrate Overlay Frame\nActive: {frame_names[0]}", fontweight='bold')

    ax_slider_sub = plt.axes([0.15, 0.12, 0.70, 0.03])
    slider_sub = Slider(ax=ax_slider_sub, label='Substrate ', valmin=0, valmax=total_frames-1, valinit=0, valfmt='%0.0f')

    def update_sub_preview(val):
        idx = int(slider_sub.val)
        line_sub.set_ydata(raw_matrix[:, idx])
        ax_sub.set_ylim(np.nanmin(raw_matrix[:, idx])*0.95, np.nanmax(raw_matrix[:, idx])*1.05)
        fig_sub.suptitle(f"Select Substrate Overlay Frame\nActive: {frame_names[idx]}", fontweight='bold')
        fig_sub.canvas.draw_idle()

    slider_sub.on_changed(update_sub_preview)
    ax_btn_sub = plt.axes([0.42, 0.03, 0.16, 0.05])
    btn_confirm_sub = Button(ax=ax_btn_sub, label='Lock Overlay', color='indianred', hovercolor='lightcoral')

    def lock_substrate(event):
        global has_passive_substrate, substrate_vector
        idx = int(slider_sub.val)
        substrate_vector = raw_matrix[:, idx]
        has_passive_substrate = True
        plt.close(fig_sub)

    btn_confirm_sub.on_clicked(lock_substrate)
    plt.show()

# =========================================================================
# PRIMARY DATA FRAME SELECTOR SCREEN
# =========================================================================
print("\n[ACTION REQUIRED] Select the data frame you wish to evaluate...")
selected_frame_idx = 0

fig_sel, ax_sel = plt.subplots(figsize=(10, 4))
plt.subplots_adjust(bottom=0.25)
line_sel, = ax_sel.plot(x_vals, raw_matrix[:, 0], '-', color='black', alpha=0.7, lw=1.5)
ax_sel.set_xlim(x_vals.min(), x_vals.max())
ax_sel.set_ylim(np.nanmin(raw_matrix[:, 0])*0.95, np.nanmax(raw_matrix[:, 0])*1.05)
ax_sel.set_xlabel(axis_label)
ax_sel.set_ylabel('Intensity [a.u.]')
fig_sel.suptitle(f"Select Active Processing Frame\nActive: {frame_names[0]}", fontweight='bold')

ax_slider_sel = plt.axes([0.15, 0.12, 0.70, 0.03])
slider_sel = Slider(ax=ax_slider_sel, label='Data Frame ', valmin=0, valmax=total_frames-1, valinit=0, valfmt='%0.0f')

def update_preview(val):
    idx = int(slider_sel.val)
    y_preview = raw_matrix[:, idx]
    line_sel.set_ydata(y_preview)
    ax_sel.set_ylim(np.nanmin(y_preview)*0.95, np.nanmax(y_preview)*1.05)
    fig_sel.suptitle(f"Select Active Processing Frame\nActive: {frame_names[idx]}", fontweight='bold')
    fig_sel.canvas.draw_idle()

slider_sel.on_changed(update_preview)
ax_btn = plt.axes([0.42, 0.03, 0.16, 0.05])
btn_confirm = Button(ax=ax_btn, label='Lock In Data', color='gold', hovercolor='khaki')

def lock_frame(event):
    global selected_frame_idx
    selected_frame_idx = int(slider_sel.val)
    plt.close(fig_sel)

btn_confirm.on_clicked(lock_frame)
plt.show()

y_vals = raw_matrix[:, selected_frame_idx]
clean_mask = ~np.isnan(y_vals)
x_clean = x_vals[clean_mask]
y_clean = y_vals[clean_mask]

bg_clean = calculate_snip_background(y_clean, iterations=SNIP_ITERATIONS)
bg_curve = np.zeros_like(y_vals)
bg_curve[clean_mask] = bg_clean
bg_curve[~clean_mask] = np.nan

y_subtracted_clean = y_clean - bg_clean

# =========================================================================
# PEAK SELECTION MATRIX
# =========================================================================
if IS_ITO_CALIBRATION:
    print("[INSTRUCTION] Left-click peak apexes to fit, then CLOSE window.")
    manual_centers = []
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            manual_centers.append(event.xdata)
            ax_pick.plot(event.xdata, event.ydata, 'X', color='crimson', markersize=10)
            fig_pick.canvas.draw()
    fig_pick, ax_pick = plt.subplots(figsize=(12, 5))
    ax_pick.plot(x_clean, y_subtracted_clean, '-', color='blue', lw=1.5)
    if len(ITO_CENTERS) > 0:
        for ref_center in ITO_CENTERS:
            ax_pick.axvline(ref_center, color='green', linestyle='--', alpha=0.4)
    fig_pick.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    if len(manual_centers) == 0: sys.exit("No calibration points chosen.")
    detected_peaks = manual_centers
else:
    all_peaks = detect_peaks_robust(y_subtracted_clean, x_clean, prominence_factor=0.015, min_distance=10)
    detected_peaks = []
    target_peak_data = {}

    for peak_idx in all_peaks:
        q_peak = x_clean[peak_idx]
        for peak_name, target_info in TARGET_PEAKS.items():
            q_min, q_max = target_info['q_range']
            if q_min <= q_peak <= q_max:
                target_peak_data[peak_name] = {'index': peak_idx, 'q': q_peak, 'intensity': y_subtracted_clean[peak_idx]}
                detected_peaks.append(q_peak)
                break
        else:
            if y_subtracted_clean[peak_idx] > np.max(y_subtracted_clean) * 0.08:
                detected_peaks.append(q_peak)

    if '100' not in target_peak_data:
        low_threshold_peaks, _ = find_peaks(y_subtracted_clean, prominence=np.max(y_subtracted_clean) * 0.003, distance=10)
        for peak_idx in low_threshold_peaks:
            q_peak = x_clean[peak_idx]
            if 0.9 <= q_peak <= 1.15:
                target_peak_data['100'] = {'index': peak_idx, 'q': q_peak, 'intensity': y_subtracted_clean[peak_idx]}
                if q_peak not in detected_peaks: detected_peaks.append(q_peak)
                break

# =========================================================================
# STAGE 2: MATHEMATICAL PEAK PROFILE OPTIMIZATION
# =========================================================================
print("\n[GLOBAL FITTING]")
initial_guesses = []
lower_bounds = []
upper_bounds = []

for center_guess in detected_peaks:
    idx_near = np.abs(x_clean - center_guess).argmin()
    amp_guess = y_subtracted_clean[idx_near]

    if IS_ITO_CALIBRATION:
        initial_guesses.extend([amp_guess, center_guess, 0.02, 0.5])
        lower_bounds.extend([0.0, center_guess - 0.12, 0.004, 0.0])
        upper_bounds.extend([np.inf, center_guess + 0.12, 0.12, 1.0])
    else:
        expected_inst_fwhm = max(0.005, fwhm_irf_curve(center_guess))
        expected_inst_eta = np.clip(eta_irf_curve(center_guess), 0.0, 1.0)

        # --- ADJUSTED WIDTH FLOOR ---
        # Dropped static flooring scaling constraints to prevent delta/sharp peaks from locking out
        fwhm_min = 0.001
        fwhm_max = max(0.20, expected_inst_fwhm * 10.0)

        initial_guesses.extend([amp_guess, center_guess, expected_inst_fwhm * 1.2, expected_inst_eta])
        lower_bounds.extend([0.0, center_guess - 0.08, fwhm_min, 0.0])
        upper_bounds.extend([np.inf, center_guess + 0.08, fwhm_max, 1.0])

initial_guesses.extend([0.0, 0.0])
lower_bounds.extend([-np.inf, -np.inf])
upper_bounds.extend([np.inf, np.inf])

fitted_y = np.zeros_like(x_vals)
r2_value = 0.0
popt = []

if len(initial_guesses) > 2:
    try:
        # --- FIX B: UNBIASED UNIFORM WEIGHTING ---
        # Switched sigma to None. This stops the optimizer from artificially treating baseline
        # points as more important than sharp peak apex pixels.
        popt, _ = curve_fit(
            multi_pv_model, x_clean, y_subtracted_clean,
            p0=initial_guesses, bounds=(lower_bounds, upper_bounds),
            sigma=None, absolute_sigma=False,
            maxfev=60000, ftol=1e-11, xtol=1e-11
        )
        fitted_y = multi_pv_model(x_vals, *popt)
        fit_clean = multi_pv_model(x_clean, *popt)

        residual_sum_squares = np.sum((y_subtracted_clean - fit_clean) ** 2)
        total_sum_squares = np.sum((y_subtracted_clean - np.mean(y_subtracted_clean)) ** 2)
        if total_sum_squares > 0:
            r2_value = 1.0 - (residual_sum_squares / total_sum_squares)
        print(f"--> Optimization Completed. Improved R² = {r2_value:.6f}")
    except Exception as e:
        print(f"[FIT CRASH] Optimization error: {e}")

# =========================================================================
# UPDATE CALIBRATION IF IN CALIBRATION MODE
# =========================================================================
if IS_ITO_CALIBRATION and len(popt) > 0:
    new_centers = popt[1:-2:4]
    new_fwhms = popt[2:-2:4]
    new_etas = popt[3:-2:4]

    sorted_indices = np.argsort(new_centers)
    sorted_centers = new_centers[sorted_indices]
    sorted_fwhms = new_fwhms[sorted_indices]
    sorted_etas = np.clip(new_etas[sorted_indices], 0.0, 1.0)

    save_calibration(sorted_centers.tolist(), sorted_fwhms.tolist(), sorted_etas.tolist())

    print("\n" + "="*70)
    print("   NEW ITO CALIBRATION VALUES GENERATED")
    print("   (Copy and paste these directly into the top of your script!)")
    print("="*70)
    print(f"DEFAULT_ITO_CENTERS = [{', '.join(f'{x:.5f}' for x in sorted_centers)}]")
    print(f"DEFAULT_ITO_FWHMS   = [{', '.join(f'{x:.5f}' for x in sorted_fwhms)}]")
    print(f"DEFAULT_ITO_ETAS    = [{', '.join(f'{x:.5f}' for x in sorted_etas)}]")
    print("="*70 + "\n")

# =========================================================================
# STAGE 4: ADVANCED DECONVOLUTION (SYNCHROTRON-OPTIMIZED SCHERRER ANALYSIS)
# =========================================================================
scherrer_results = {}

IRF_SCALE = 1.00
MICROSTRAIN_PCT = 0.00  # 0.0 decoupled baseline for clean synchrotron resolution sweeps

# --- SYNCHROTRON CONFIGURATION ---
# Bypasses the pre-broadened nanocrystalline substrate profile when scaling sample targets
USING_SYNCHROTRON_DIRECT_IRF = True
SYNCHROTRON_FWHM_FLOOR = 0.003       # Typical analytical resolution limit floor in q for ALS 12.3

if not IS_ITO_CALIBRATION and len(popt) > 2:
    fitted_centers = popt[1:-2:4]
    fitted_fwhms = popt[2:-2:4]

    for peak_name, target_info in TARGET_PEAKS.items():
        q_min, q_max = target_info['q_range']
        best_match_idx = None
        min_dist = np.inf

        for idx, center in enumerate(fitted_centers):
            if q_min <= center <= q_max:
                dist = abs(center - target_info['q']) if 'q' in target_info else 0
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = idx

        if best_match_idx is not None:
            cen = fitted_centers[best_match_idx]
            fwhm_measured = fitted_fwhms[best_match_idx]

            # Route the deconvolution logic through the high-resolution optics override
            if USING_SYNCHROTRON_DIRECT_IRF:
                fwhm_inst = SYNCHROTRON_FWHM_FLOOR
            else:
                fwhm_inst = fwhm_irf_curve(cen) * IRF_SCALE

            strain_factor = MICROSTRAIN_PCT / 100.0
            fwhm_strain = strain_factor * cen

            fwhm_total_distortion = np.sqrt(fwhm_inst**2 + fwhm_strain**2)

            # Deep synchrotron physical resolution floor
            ABS_RESOLUTION_FLOOR = 0.0025

            if fwhm_measured > fwhm_total_distortion:
                beta_size = np.sqrt(fwhm_measured**2 - fwhm_total_distortion**2)
                is_limited = False
            else:
                beta_size = ABS_RESOLUTION_FLOOR
                is_limited = True

            if beta_size < ABS_RESOLUTION_FLOOR:
                beta_size = ABS_RESOLUTION_FLOOR
                is_limited = True

            domain_size_A = (2.0 * np.pi * K_SHAPE_FACTOR) / beta_size
            domain_size_nm = domain_size_A / 10.0

            scherrer_results[peak_name] = {
                'center': cen,
                'measured_fwhm': fwhm_measured,
                'inst_fwhm': fwhm_inst,
                'size_nm': domain_size_nm,
                'is_limited': is_limited
            }

if scherrer_results and not IS_ITO_CALIBRATION:
    print("\n" + "-"*55)
    print("   SCHERRER CRYSTALLITE SIZE SUMMARY")
    print("-" * 55)
    for peak, data in scherrer_results.items():
        if data['is_limited']:
            print(f" Peak ({peak}) at q={data['center']:.3f} Å⁻¹ -> Size: ~{data['size_nm']:.1f} nm (Instrument Floor Restrained)")
        else:
            print(f" Peak ({peak}) at q={data['center']:.3f} Å⁻¹ -> Size: {data['size_nm']:.1f} nm")
    print("-" * 55)

# =========================================================================
# STRUCTURAL REFLECTION ENGINE MAPPING
# =========================================================================
valid_reflections = []
if run_structural_hkl and not IS_ITO_CALIBRATION and len(popt) > 2 and a_mapi is not None:
    fitted_centers = popt[1:-2:4]

    possible_hkls = []
    for h in range(-3, 4):
        for k in range(-3, 4):
            for l in range(-3, 4):
                if h == 0 and k == 0 and l == 0: continue
                sq = h**2 + k**2 + l**2
                if sq <= 12:
                    fam = tuple(sorted((abs(h), abs(k), abs(l)), reverse=True))
                    if fam not in possible_hkls: possible_hkls.append(fam)

    for h, k, l in possible_hkls:
        sq = h**2 + k**2 + l**2
        q_ref = (2 * np.pi * np.sqrt(sq)) / a_mapi
        if Q_MIN <= q_ref <= Q_MAX:
            valid_reflections.append({'label': f"({h}{k}{l})", 'q_mapi': q_ref})

# =========================================================================
# FINAL LAYOUT RENDER ENGINE
# =========================================================================
fig_fit = plt.figure(figsize=(15, 11))
gs = fig_fit.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[3, 1.2, 1.5], hspace=0.35, wspace=0.18)

ax1 = fig_fit.add_subplot(gs[0, 0])
ax2 = fig_fit.add_subplot(gs[0, 1])
ax3 = fig_fit.add_subplot(gs[1, 0], sharex=ax1)
ax4 = fig_fit.add_subplot(gs[1, 1], sharex=ax2)
ax5 = fig_fit.add_subplot(gs[2, :])

# --- PANEL 1 ---
ax1.plot(x_vals, y_vals, 'o', ms=2, color='black', alpha=0.4, label='Raw Film Data')
ax1.plot(x_vals, bg_curve, '-', color='darkorange', lw=1.8, label='SNIP Baseline')
if has_passive_substrate:
    ax1.plot(x_vals, substrate_vector, ':', color='crimson', lw=1.2, alpha=0.7, label='ITO Reference')
ax1.set_title("1. Raw Integration & Background", fontsize=11, fontweight='bold')
ax1.set_ylabel('Intensity [a.u.]', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# --- PANEL 2 ---
ax2.plot(x_vals, y_vals - bg_curve, '-', color='dimgray', alpha=0.4, lw=1, label='Subtracted Baseline')
if len(popt) > 2:
    ax2.plot(x_vals, fitted_y, '-', color='crimson', lw=2.0, label=f'Global Fit (R²={r2_value:.4f})')
    peak_centers = popt[1:-2:4]
    for c in peak_centers:
        ax2.axvline(c, color='blue', linestyle=':', alpha=0.6)
ax2.set_title("2. Mathematical Multi-Peak Fitting Space", fontsize=11, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# --- PANEL 3 ---
if len(popt) > 2:
    res_clean = y_subtracted_clean - fit_clean
    ax3.plot(x_clean, res_clean, '-', color='purple', lw=1, label='Residuals')
    ax3.axhline(0, color='black', lw=1, ls='--')
ax3.set_title("3. Left-Channel Residual Matrix", fontsize=10, fontweight='bold')
ax3.set_xlabel(axis_label, fontweight='bold')
ax3.set_ylabel('Δ Intensity', fontweight='bold')
ax3.grid(True, alpha=0.3)

# --- PANEL 4 ---
if run_structural_hkl and not IS_ITO_CALIBRATION:
    ax4.plot(x_vals, fitted_y, '-', color='crimson', alpha=0.5)
    for ref in valid_reflections:
        ax4.axvline(ref['q_mapi'], color='teal', alpha=0.7, lw=1.5)
        ax4.text(ref['q_mapi'], np.max(fitted_y)*0.7, ref['label'], rotation=90, color='teal', fontsize=8)
ax4.set_title("4. Crystallographic Reference Overlays", fontsize=10, fontweight='bold')
ax4.set_xlabel(axis_label, fontweight='bold')
ax4.grid(True, alpha=0.3)

# --- PANEL 5: SCHERRER ANALYSIS BAR CHART ---
if scherrer_results and not IS_ITO_CALIBRATION:
    peaks = list(scherrer_results.keys())
    sizes = [data['size_nm'] for data in scherrer_results.values()]
    q_positions = [data['center'] for data in scherrer_results.values()]

    labels = [f"({p})\n$q$ = {q:.2f} $\\AA^{{-1}}$" for p, q in zip(peaks, q_positions)]

    bar_colors = ['#d9534f' if scherrer_results[p]['is_limited'] else 'royalblue' for p in peaks]
    edge_colors = ['darkred' if scherrer_results[p]['is_limited'] else 'midnightblue' for p in peaks]

    bars = ax5.barh(labels, sizes, color=bar_colors, edgecolor=edge_colors, height=0.5, alpha=0.85)
    ax5.set_xlabel("Mean Domain Crystallite Size (nm)", fontweight='bold')
    ax5.set_title("5. Calculated Perovskite Crystallite Domain Sizes (Scherrer Bound)", fontsize=11, fontweight='bold')
    ax5.set_xlim(0, 250)  # Expanded viewing threshold for high-resolution setups
    ax5.grid(True, axis='x', linestyle='--', alpha=0.5)

    for bar, p in zip(bars, peaks):
        width = bar.get_width()
        if scherrer_results[p]['is_limited']:
            text_disp = f"~{width:.1f} nm\n(Upper Sync-Limit Floor)"
            text_color = 'darkred'
        else:
            text_disp = f'{width:.1f} nm'
            text_color = 'navy'

        ax5.text(width + 1.5, bar.get_y() + bar.get_height()/2,
                 text_disp, va='center', ha='left', fontweight='bold', color=text_color, fontsize=9)
else:
    msg = "ITO Calibration active. Synchrotron baseline resolution index generated." if IS_ITO_CALIBRATION else "No target perovskite phase peaks (100) detected."
    ax5.text(0.5, 0.5, msg, ha='center', va='center', fontsize=11, color='crimson', fontstyle='italic')
    ax5.set_title("5. Calculated Perovskite Crystallite Domain Sizes (Scherrer)", fontsize=11, fontweight='bold')

plt.show()
