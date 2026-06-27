## lastly, you will use the "test_scherrer.py" script to run the Scherrer analysis on the 100 peak across all frames in your dataset. This will allow you to track the growth of the perovskite grains over time and export the data to a CSV file for further analysis and plotting.
# You must pull the centeres and FWHM values from the terminal output of the "peak isolation for tifs.py" script and update the calibration parameters in the "test_scherrer.py" script before running it to get accurate results.

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog, ttk
import tkinter.messagebox as messagebox

import matplotlib
matplotlib.use('TkAgg')
plt.style.use('seaborn-v0_8-whitegrid')

# =========================================================================
# X-RAY CONVERSION CONSTANTS
# =========================================================================
ENERGY_KEV = 10.0
WAVELENGTH_A = 12.3984 / ENERGY_KEV
K_SHAPE_FACTOR = 0.90

GLOBAL_CALIBRATION_FILE = "ito_calibration.npz"

def load_calibration():
    """Load calibration data from file or use hardcoded defaults if missing"""
    if os.path.exists(GLOBAL_CALIBRATION_FILE):
        try:
            data = np.load(GLOBAL_CALIBRATION_FILE)
            return data['centers'].tolist(), data['fwhms'].tolist()
        except:
            pass
    return [1.52301, 2.14004, 2.46856, 3.48597, 4.09467],[0.02173, 0.03256, 0.03200, 0.04500, 0.04867]#these are your default values from the terminal output of the "peak isolation for tifs.py" script, make sure to update these with your new values if you ran the calibration refinement in that script to get more accurate results in this one

ITO_CENTERS, ITO_FWHMS = load_calibration()

def twotheta_to_q(twotheta_deg, wavelength):
    return (4.0 * np.pi / wavelength) * np.sin(np.radians(twotheta_deg) / 2.0)

# Build the Instrumental Resolution Function Curve
fwhm_irf_curve = np.poly1d(np.polyfit(ITO_CENTERS, ITO_FWHMS, deg=2 if len(ITO_CENTERS) >=3 else 1))

# =========================================================================
# DEFINING TARGET NARROW-PEAK RANGES (Strictly isolating the 100)
# =========================================================================
Q_MIN = 0.85
Q_MAX = 1.20   # Lowered upper bound to focus computing resources solely on 100 peak
SNIP_ITERATIONS = 12

TARGET_PEAKS = {
    '100': {'q_range': (0.90, 1.15)},
    # '110': {'q_range': (1.35, 1.55)},  # Commented out to focus on 100
    # '111': {'q_range': (1.65, 1.85)}   # Commented out to focus on 100
}

def select_file_via_popup(title_text, file_types=[("CSV Files", "*.csv")]):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(title=title_text, filetypes=file_types)
    root.destroy()
    return Path(file_path) if file_path else None

def save_file_via_popup(title_text, default_name="scherrer_kinetics_output.csv"):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.asksaveasfilename(
        title=title_text,
        initialfile=default_name,
        filetypes=[("CSV Files", "*.csv")],
        defaultextension=".csv"
    )
    root.destroy()
    return Path(file_path) if file_path else None

# =========================================================================
# INTERACTIVE RANGE SELECTION POPUP GUI
# =========================================================================
def get_frame_range_cutoff(max_frames):
    root = tk.Tk()
    root.title("Select Frame Processing Range")
    root.attributes('-topmost', True)
    root.geometry("450x220")

    start_var = tk.IntVar(value=0)
    end_var = tk.IntVar(value=max_frames - 1)
    confirmed = tk.BooleanVar(value=False)

    ttk.Label(root, text="Select Frame Window to Analyze", font=('Helvetica', 12, 'bold')).pack(pady=10)
    lbl_status = ttk.Label(root, text=f"Processing Frames: 0 to {max_frames - 1}", font=('Helvetica', 10, 'italic'))

    def update_labels(*args):
        s = start_var.get()
        e = end_var.get()
        if s >= e:
            s = e - 1
            start_var.set(s)
        lbl_status.config(text=f"Processing Window: Frame {s} to Frame {e} (Total: {e - s + 1} frames)")

    frame_s = ttk.Frame(root)
    frame_s.pack(fill='x', padx=20, pady=5)
    ttk.Label(frame_s, text="Start Frame: ", width=12).pack(side='left')
    slider_start = ttk.Scale(frame_s, from_=0, to=max_frames-1, variable=start_var, orient='horizontal', command=update_labels)
    slider_start.pack(side='left', fill='x', expand=True)

    frame_e = ttk.Frame(root)
    frame_e.pack(fill='x', padx=20, pady=5)
    ttk.Label(frame_e, text="End Frame: ", width=12).pack(side='left')
    slider_end = ttk.Scale(frame_e, from_=0, to=max_frames-1, variable=end_var, orient='horizontal', command=update_labels)
    slider_end.pack(side='left', fill='x', expand=True)

    lbl_status.pack(pady=10)

    def on_confirm():
        confirmed.set(True)
        root.destroy()

    ttk.Button(root, text="Lock Window & Run Fit", command=on_confirm).pack(pady=10)
    update_labels()
    root.mainloop()

    if confirmed.get():
        return start_var.get(), end_var.get()
    else:
        sys.exit("Operation cancelled by user.")

def calculate_snip_background(y_data, iterations=12):
    bg = np.copy(y_data)
    n = len(y_data)
    for p in range(1, iterations + 1):
        if 2*p >= n: break
        left, right = bg[0 : n - 2*p], bg[2*p : n]
        avg = (left + right) / 2.0
        bg[p : n - p] = np.where(bg[p : n - p] > avg, avg, bg[p : n - p])
    return bg

def pseudo_voigt(x, amp, cen, fwhm, eta=0.5):
    sigma = fwhm / 2.35482
    g_profile = np.exp(-((x - cen)**2) / (2.0 * sigma**2))
    gamma = fwhm / 2.0
    l_profile = (gamma**2) / ((x - cen)**2 + gamma**2)
    return amp * ((eta * l_profile) + ((1.0 - eta) * g_profile))

def single_pv_model(x, amp, cen, fwhm, eta, c0, c1):
    return c0 + c1 * x + pseudo_voigt(x, amp, cen, fwhm, eta)

# =========================================================================
# DATA LOADING
# =========================================================================
print("Select your compiled 1D integrated dataset (.csv)...")
cache_csv_path = select_file_via_popup("Select Integrated 1D Dataset CSV")
if not cache_csv_path: sys.exit("No data file selected.")

raw_df = pd.read_csv(cache_csv_path)

time_strings = list(raw_df.columns[1:])
try:
    time_values = [float(t) for t in time_strings]
    time_axis_label = "Time"
except ValueError:
    time_values = list(range(len(time_strings)))
    time_axis_label = "Frame Index"

cleaned_df = raw_df.iloc[1:].copy()
raw_twotheta = pd.to_numeric(cleaned_df.iloc[:, 0]).values

cut_mask = raw_twotheta >= 5.0
x_vals_pre = twotheta_to_q(raw_twotheta[cut_mask], WAVELENGTH_A)

q_clip_mask = (x_vals_pre >= Q_MIN) & (x_vals_pre <= Q_MAX)
x_global = x_vals_pre[q_clip_mask]

raw_matrix = cleaned_df.iloc[:, 1:].apply(pd.to_numeric).values[cut_mask, :][q_clip_mask, :]
total_frames_available = raw_matrix.shape[1]

start_frame, end_frame = get_frame_range_cutoff(total_frames_available)

raw_matrix = raw_matrix[:, start_frame:end_frame+1]
time_values = np.array(time_values)[start_frame:end_frame+1]
total_frames = raw_matrix.shape[1]

# Master storage lists specifically for compiling the final exportable data rows
export_rows = []

history_size = []
history_r2 = []
history_time = []
history_fwhm_meas = []
history_center = []

# =========================================================================
# BATCH KINETICS PROCESSING LOOP (100 EXCLUSIVE)
# =========================================================================
print(f"\n[STARTING ISOLATED BATCH FIT] Processing frames {start_frame} to {end_frame}...")

for idx in range(total_frames):
    y_global = raw_matrix[:, idx]
    if np.any(np.isnan(y_global)): continue

    bg_global = calculate_snip_background(y_global, iterations=SNIP_ITERATIONS)
    y_subtracted_global = y_global - bg_global

    # Process just the single '100' peak left in the dictionary
    for peak_name, info in TARGET_PEAKS.items():
        q_start, q_end = info['q_range']

        sub_mask = (x_global >= q_start) & (x_global <= q_end)
        x_sub = x_global[sub_mask]
        y_sub = y_subtracted_global[sub_mask]

        if len(y_sub) < 5: continue

        peak_idx = np.argmax(y_sub)
        cen_guess = x_sub[peak_idx]
        amp_guess = y_sub[peak_idx]

        expected_fwhm = fwhm_irf_curve(cen_guess)
        fwhm_lower_limit = expected_fwhm * 1.01

        p0 = [amp_guess, cen_guess, expected_fwhm * 1.8, 0.5, 0.0, 0.0]
        bounds = (
            [amp_guess * 0.1, cen_guess - 0.03, fwhm_lower_limit, 0.0, -np.inf, -np.inf],
            [amp_guess * 10.0, cen_guess + 0.03, 0.25, 1.0, np.inf, np.inf]
        )

        try:
            popt, _ = curve_fit(single_pv_model, x_sub, y_sub, p0=p0, bounds=bounds, maxfev=20000)
            fit_y = single_pv_model(x_sub, *popt)

            ss_res = np.sum((y_sub - fit_y) ** 2)
            ss_tot = np.sum((y_sub - np.mean(y_sub)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0

            if r2 > 0.80 and popt[0] > (np.max(y_subtracted_global) * 0.015):
                cen_fit = popt[1]
                fwhm_measured = popt[2]

                fwhm_inst = fwhm_irf_curve(cen_fit)
                fwhm_distortion = fwhm_inst * 1.2
                ABS_FLOOR = 0.007

                if fwhm_measured > fwhm_distortion:
                    beta_size = np.sqrt(fwhm_measured**2 - fwhm_distortion**2)
                else:
                    beta_size = ABS_FLOOR

                beta_size = max(beta_size, ABS_FLOOR)
                domain_size_nm = ((2.0 * np.pi * K_SHAPE_FACTOR) / beta_size) / 10.0

                # Append data points to the plot lists
                current_time = time_values[idx]
                history_size.append(domain_size_nm)
                history_r2.append(r2)
                history_time.append(current_time)
                history_fwhm_meas.append(fwhm_measured)
                history_center.append(cen_fit)

                # Append comprehensive tracking metadata row for CSV compilation
                export_rows.append({
                    'Frame_Index': start_frame + idx,
                    'Time_Value': current_time,
                    'Peak_Assignment': '100',
                    'Fitted_Center_q_A^-1': cen_fit,
                    'Measured_FWHM_q': fwhm_measured,
                    'Instrument_FWHM_q': fwhm_inst,
                    'Pure_Structural_Broadening_beta': beta_size,
                    'Scherrer_Domain_Size_nm': domain_size_nm,
                    'Fit_Quality_R2': r2
                })
        except:
            continue

print("[COMPLETE] High-precision isolated tracking phase complete.")

# =========================================================================
# TIME-SERIES VISUALIZATION
# =========================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1.2])

if history_time:
    # Top Panel: Domain Growth curve
    color = 'royalblue'
    ax1.plot(history_time, history_size, '-o', color=color, ms=5, lw=2, label='(100) Domain Size')
    ax1.set_ylabel("Crystallite Size $D_{100}$ (nm)", color=color, fontweight='bold', fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max(history_size) * 1.15)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title(f"Isolated (100) Perovskite Grain Growth Kinetics\nProcessing Frames {start_frame} to {end_frame}", fontsize=12, fontweight='bold')

    # Bottom Panel: Fit Confidence Quality R^2 path
    color = 'darkorange'
    ax2.plot(history_time, history_r2, ':', color=color, marker='o', ms=3, alpha=0.7, label='Fit Quality ($R^2$)')
    ax2.set_xlabel(f"{time_axis_label}", fontweight='bold', fontsize=11)
    ax2.set_ylabel("Fit Quality ($R^2$)", color=color, fontweight='bold', fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.75, 1.02)
    ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
print("\n[DISPLAY] Rendering interactive kinetic growth timeline plot window. Close window to initiate data export...")
plt.show()

# =========================================================================
# FILE SAVE/EXPORT ROUTINE
# =========================================================================
if export_rows:
    print("\n[EXPORT ROUTINE] Compiling data array...")
    export_df = pd.DataFrame(export_rows)

    print("Launching Save File dialog prompt...")
    save_path = save_file_via_popup("Save Extracted Scherrer Kinetics Data CSV")

    if save_path:
        export_df.to_csv(save_path, index=False)
        print(f"\n" + "="*60)
        print(f" SUCCESS! Data saved seamlessly to:")
        print(f" {save_path}")
        print("="*60 + "\n")
    else:
        print("\n[WARNING] Export operation aborted. No filename was declared.")
else:
    print("\n[ERROR] No reliable mathematical profiles were calculated, skipping data file export.")
