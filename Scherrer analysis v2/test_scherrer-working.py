import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog

# =========================================================================
# CONSTANTS & CALIBRATION SETUP
# =========================================================================
ENERGY_KEV = 10.0
WAVELENGTH_A = 12.3984 / ENERGY_KEV
K_SHAPE_FACTOR = 0.90
SNIP_ITERATIONS = 12

# Synchrotron Deconvolution Floor Settings
USING_SYNCHROTRON_DIRECT_IRF = True
SYNCHROTRON_FWHM_FLOOR = 0.003
ABS_RESOLUTION_FLOOR = 0.0025

# HARDCODED AUTOMATED WINDOW FOR THE CUBIC MAPI (100) PEAK
Q_MIN, Q_MAX = 0.90, 1.15

# HARDCODED MAX END FRAME CONSTRAINT
MAX_END_FRAME = 275

# =========================================================================
# CORE MATH FUNCTIONS
# =========================================================================
def twotheta_to_q(twotheta_deg, wavelength):
    return (4.0 * np.pi / wavelength) * np.sin(np.radians(twotheta_deg) / 2.0)

def calculate_snip_background(y_data, iterations=12):
    bg = np.copy(y_data)
    n = len(y_data)
    if np.any(np.isnan(bg)):
        x = np.arange(n)
        mask = ~np.isnan(bg)
        if np.sum(mask) > 1: bg = np.interp(x, x[mask], bg[mask])
        else: bg = np.zeros_like(bg)

    for p in range(1, iterations + 1):
        if 2*p >= n: break
        left, right = bg[0 : n - 2*p], bg[2*p : n]
        avg = (left + right) / 2.0
        bg[p : n - p] = np.where(bg[p : n - p] > avg, avg, bg[p : n - p])
    return bg

def pseudo_voigt(x, amp, cen, fwhm, eta=0.5):
    if fwhm <= 0: fwhm = 0.001
    sigma = fwhm / 2.35482
    g_profile = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((x - cen)**2) / (2.0 * sigma**2))
    g_profile /= (1.0 / (sigma * np.sqrt(2.0 * np.pi)))
    gamma = fwhm / 2.0
    l_profile = (1.0 / np.pi) * (gamma / ((x - cen)**2 + gamma**2))
    l_profile /= (1.0 / (np.pi * gamma))
    return amp * ((eta * l_profile) + ((1.0 - eta) * g_profile))

def single_pv_model(x, amp, cen, fwhm, eta, c0, c1):
    return c0 + c1 * x + pseudo_voigt(x, amp, cen, fwhm, eta)

# =========================================================================
# FILE SELECTION
# =========================================================================
root = tk.Tk()
root.withdraw()

print("Select your compiled 1D integrated dataset (.csv)...")
csv_path = filedialog.askopenfilename(title="Select Integrated 1D Dataset CSV", filetypes=[("CSV Files", "*.csv")])
if not csv_path: sys.exit("No file selected.")

raw_df = pd.read_csv(csv_path)
time_headers = list(raw_df.columns[1:])
total_raw_frames = len(time_headers)

# Apply absolute hard stop constraint at frame 275
end_frame_limit = min(MAX_END_FRAME, total_raw_frames)

try:
    time_numeric = [float(t.split()[0]) for t in time_headers]
    time_label = "Time / Frame Metric"
except:
    time_numeric = list(range(len(time_headers)))
    time_label = "Frame Index"

cleaned_df = raw_df.iloc[1:].copy()
raw_twotheta = pd.to_numeric(cleaned_df.iloc[:, 0]).values

cut_mask = raw_twotheta >= 5.0
x_vals_pre = twotheta_to_q(raw_twotheta[cut_mask], WAVELENGTH_A)
q_clip_mask = (x_vals_pre >= Q_MIN) & (x_vals_pre <= Q_MAX)
x_vals = x_vals_pre[q_clip_mask]

raw_matrix = cleaned_df.iloc[:, 1:].apply(pd.to_numeric).values[cut_mask, :][q_clip_mask, :]

# =========================================================================
# BATCH PROCESSING WITH AUTOMATIC NOISE FILTERING & MAX LIMIT
# =========================================================================
print(f"\n[BATCH] Automated MAPI (100) peak processing running from Frame 1 to {end_frame_limit}...")
print(f"--> Target window fixed automatically at: {Q_MIN:.2f} - {Q_MAX:.2f} Å⁻¹")

extracted_times = []
domain_sizes_nm = []
peak_centers_q = []
measured_fwhms = []
frame_indices = []

# Estimate global matrix noise threshold to drop dead baseline scans automatically
global_max_amplitude = np.nanmax(raw_matrix[:, :end_frame_limit])

for idx in range(end_frame_limit):
    y_vals = raw_matrix[:, idx]
    clean_mask = ~np.isnan(y_vals)
    if np.sum(clean_mask) < 10: continue

    x_clean = x_vals[clean_mask]
    y_clean = y_vals[clean_mask]

    bg_clean = calculate_snip_background(y_clean, iterations=SNIP_ITERATIONS)
    y_sub = y_clean - bg_clean

    peak_idx_local = np.argmax(y_sub)
    q_guess = x_clean[peak_idx_local]
    amp_guess = y_sub[peak_idx_local]

    # AUTOMATIC FRAME FILTER: Drops early noisy frames if peak isn't defined yet
    if amp_guess < (global_max_amplitude * 0.02):
        continue

    initial_guesses = [amp_guess, q_guess, 0.015, 0.5, 0.0, 0.0]
    lower_bounds = [0.0, q_guess - 0.04, 0.001, 0.0, -np.inf, -np.inf]
    upper_bounds = [np.inf, q_guess + 0.04, 0.15, 1.0, np.inf, np.inf]

    try:
        popt, _ = curve_fit(
            single_pv_model, x_clean, y_sub,
            p0=initial_guesses, bounds=(lower_bounds, upper_bounds),
            sigma=None, maxfev=10000
        )

        fwhm_measured = popt[2]
        cen = popt[1]

        if not (Q_MIN <= cen <= Q_MAX):
            continue

        fwhm_inst = SYNCHROTRON_FWHM_FLOOR
        if fwhm_measured > fwhm_inst:
            beta_size = np.sqrt(fwhm_measured**2 - fwhm_inst**2)
        else:
            beta_size = ABS_RESOLUTION_FLOOR

        if beta_size < ABS_RESOLUTION_FLOOR:
            beta_size = ABS_RESOLUTION_FLOOR

        domain_size_nm = ((2.0 * np.pi * K_SHAPE_FACTOR) / beta_size) / 10.0

        extracted_times.append(time_numeric[idx])
        domain_sizes_nm.append(domain_size_nm)
        peak_centers_q.append(cen)
        measured_fwhms.append(fwhm_measured)
        frame_indices.append(idx + 1)

    except Exception as e:
        continue

if len(domain_sizes_nm) == 0:
    sys.exit("\n[ERROR] No valid cubic MAPI phase reflections cleared the automated filter threshold.")

start_frame = frame_indices[0]
end_frame = frame_indices[-1]

# =========================================================================
# EXPORT DATA SETUP
# =========================================================================
print("\nSelect the destination directory to export your analysis results...")
export_dir = filedialog.askdirectory(title="Select Export Directory")

if export_dir:
    export_path = Path(export_dir)

    output_df = pd.DataFrame({
        'Original_Frame_Index': frame_indices,
        time_label: extracted_times,
        'Crystallite_Domain_Size_nm': domain_sizes_nm,
        'Peak_Center_q_A_inv': peak_centers_q,
        'Measured_FWHM_q_A_inv': measured_fwhms
    })

    csv_out_name = export_path / "scherrer_kinetics_automated_output.csv"
    output_df.to_csv(csv_out_name, index=False)
    print(f"\n[SUCCESS] Exported kinetic data ({len(frame_indices)} active frames analyzed) to:\n--> {csv_out_name}")
else:
    print("[WARNING] No directory chosen. Exporting skipped.")

# =========================================================================
# GENERATE TEMPORAL EVOLUTION PLOT
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 5), layout="tight")
ax.plot(extracted_times, domain_sizes_nm, '-o', color='royalblue', lw=2, ms=4, label='MAPI (100) Domain Tracking')

ax.set_title(f"Cubic MAPI (100) Crystallite Domain Size Kinetics (Active Frames: {start_frame}-{end_frame})", fontsize=12, fontweight='bold')
ax.set_xlabel(time_label, fontweight='bold')
ax.set_ylabel("Mean Domain Size ($nm$)", fontweight='bold')
ax.grid(True, alpha=0.4)
ax.legend(loc='lower right')

if export_dir:
    fig_out_name = export_path / "scherrer_kinetics_automated_plot.png"
    plt.savefig(fig_out_name, dpi=300)
    print(f"[SUCCESS] Plot saved to:\n--> {fig_out_name}")

plt.show()
