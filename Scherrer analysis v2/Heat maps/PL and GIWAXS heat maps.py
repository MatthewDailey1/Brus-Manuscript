# -*- coding: utf-8 -*-
"""
In-situ PL & Parallelized 2D-GIWAXS Production Studio
Features interactive timeline alignment, print-ready publication exports,
1-decimal rounded tick bounds, and comprehensive CSV metadata tracking sheets.
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.widgets import Slider, RadioButtons, Button
import tkinter as tk
from tkinter import filedialog
import re
import warnings # Added for environment warning suppression
from concurrent.futures import ProcessPoolExecutor

# X-ray image processing frameworks
import fabio
import pyFAI

# Global list of publication-ready colormaps
CMAP_OPTIONS = ['plasma', 'inferno', 'viridis', 'magma', 'jet', 'gray']

def load_in_situ_pl(folder_path, max_time=275):
    if not folder_path:
        return None, None, None, ""
    current_folder = os.path.basename(os.path.normpath(folder_path))
    parent_folder  = os.path.basename(os.path.dirname(os.path.normpath(folder_path)))
    label_header   = f"{parent_folder}_{current_folder}"

    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not files:
        print(f"[-] No PL .txt files found in: {folder_path}")
        return None, None, None, ""

    files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', var)])

    sample_data = pd.read_csv(os.path.join(folder_path, files[0]), sep=r'\s+', engine='c', header=None)
    raw_axes    = pd.to_numeric(sample_data[0], errors='coerce').dropna()
    y_axis_vals = raw_axes.values
    valid_indices = raw_axes.index

    intensities, raw_timestamps = [], []
    for file in files:
        full_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(full_path, sep=r'\s+', engine='c', header=None)
            intensities.append(pd.to_numeric(df[1], errors='coerce').iloc[valid_indices].values)
            raw_timestamps.append(os.path.getmtime(full_path))
        except:
            raw_timestamps.append(len(raw_timestamps))

    data_matrix = np.array(intensities).T
    base_time = raw_timestamps[0]
    time_axis = np.array([t - base_time for t in raw_timestamps])

    if time_axis.max() == 0 or len(np.unique(time_axis)) == 1:
        time_axis = np.arange(len(files))

    return time_axis, y_axis_vals, data_matrix, label_header

def _integrate_single_frame(args):
    """Worker function executed by individual CPU worker cores."""
    # Suppress the Anaconda glymur/tiff.dll warning inside the parallel workers
    warnings.filterwarnings("ignore", category=UserWarning, module="glymur")

    full_path, poni_file_path = args
    try:
        ai = pyFAI.load(poni_file_path)
        img_frame = fabio.open(full_path).data
        res = ai.integrate1d_ng(img_frame, 500, unit="q_A^-1")
        try:
            timestamp = os.path.getmtime(full_path)
        except:
            timestamp = 0
        return res.radial, res.intensity, timestamp
    except Exception as e:
        return None, None, None

def process_parallel_giwaxs(folder_path, poni_file_path, num_cores=5):
    if not folder_path or not poni_file_path:
        return None, None, None

    valid_exts = ('.tif', '.tiff', '.edf', '.cbf')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    if not files:
        print(f"[-] No raw 2D images found in: {folder_path}")
        return None, None, None

    files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', var)])
    tasks = [(os.path.join(folder_path, f), poni_file_path) for f in files]

    intensities_1d = []
    raw_timestamps = []
    q_axis = None

    print(f"[+] Spawning {num_cores} parallel processor cores to integrate {len(files)} GIWAXS frames...")
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(_integrate_single_frame, tasks))

    for idx, (rad, inten, timestamp) in enumerate(results):
        if rad is not None:
            if q_axis is None:
                q_axis = rad
            intensities_1d.append(inten)
            raw_timestamps.append(timestamp if timestamp != 0 else idx)

    if not intensities_1d:
        return None, None, None

    giwaxs_matrix = np.array(intensities_1d).T
    base_time     = raw_timestamps[0]
    time_axis     = np.array([t - base_time for t in raw_timestamps])

    if time_axis.max() == 0 or len(np.unique(time_axis)) == 1:
        time_axis = np.arange(len(files))

    return time_axis, q_axis, giwaxs_matrix

def main():
    MAX_DISPLAY_TIME  = 275.0

    root = tk.Tk()
    root.withdraw()

    print("--- STEP 1: Select PL Folder ---")
    pl_path = filedialog.askdirectory(title="Select Folder with PL TXT Files")
    pl_time_raw, pl_energy, pl_matrix_raw, pl_label = load_in_situ_pl(pl_path, max_time=MAX_DISPLAY_TIME)
    if pl_matrix_raw is None: return

    if np.median(pl_energy) > 100: pl_energy = 1239.84193 / pl_energy
    pl_sort   = np.argsort(pl_energy)
    pl_energy = pl_energy[pl_sort]
    pl_matrix_raw = pl_matrix_raw[pl_sort, :]

    pl_mask   = (pl_energy >= 1.40) & (pl_energy <= 2.00)
    pl_energy = pl_energy[pl_mask]
    pl_matrix_raw = pl_matrix_raw[pl_mask, :]

    print("\n--- STEP 2: Select Beamline .PONI Calibration File ---")
    poni_path = filedialog.askopenfilename(title="Select Calibration .poni File", filetypes=[("PONI files", "*.poni")])
    if not poni_path: return

    print("\n--- STEP 3: Select Raw 2D GIWAXS Image Folder (.tif) ---")
    giwaxs_raw_path = filedialog.askdirectory(title="Select Folder with Raw 2D GIWAXS Images")
    giwaxs_time_raw, giwaxs_q, giwaxs_matrix_raw = process_parallel_giwaxs(giwaxs_raw_path, poni_path, num_cores=5)
    if giwaxs_matrix_raw is None: return

    gx_sort       = np.argsort(giwaxs_q)
    giwaxs_q      = giwaxs_q[gx_sort]
    giwaxs_matrix_raw = giwaxs_matrix_raw[gx_sort, :]

    giwaxs_mask   = (giwaxs_q >= 0.5) & (giwaxs_q <= 2.5)
    giwaxs_q      = giwaxs_q[giwaxs_mask]
    giwaxs_matrix_raw = giwaxs_matrix_raw[giwaxs_mask, :]

    # Alignment Offset Storage Tracking Vectors
    current_pl_shift = [0.0]
    current_giwaxs_shift = [0.0]

    # Initial time configurations bounded to active window limits
    pl_time = pl_time_raw.copy()
    pl_mask_window = (pl_time >= 0) & (pl_time <= MAX_DISPLAY_TIME)
    pl_matrix = pl_matrix_raw[:, pl_mask_window]
    pl_time = pl_time[pl_mask_window]

    giwaxs_time = giwaxs_time_raw.copy()
    giwaxs_mask_window = (giwaxs_time >= 0) & (giwaxs_time <= MAX_DISPLAY_TIME)
    giwaxs_matrix = giwaxs_matrix_raw[:, giwaxs_mask_window]
    giwaxs_time = giwaxs_time[giwaxs_mask_window]

    # =============================================================================
    # DISPLAY DASHBOARD SETUP
    # =============================================================================
    fig, (ax_pl, ax_gx) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    plt.subplots_adjust(bottom=0.25, left=0.25, hspace=0.15)

    extent_pl = [0, MAX_DISPLAY_TIME, pl_energy.min(), pl_energy.max()]
    extent_gx = [0, MAX_DISPLAY_TIME, giwaxs_q.min(), giwaxs_q.max()]

    init_gamma_pl, init_gamma_gx = 0.5, 0.5

    img_pl = ax_pl.imshow(pl_matrix, aspect='auto', cmap='plasma', origin='lower', extent=extent_pl, norm=colors.PowerNorm(gamma=init_gamma_pl))
    img_gx = ax_gx.imshow(giwaxs_matrix, aspect='auto', cmap='inferno', origin='lower', extent=extent_gx, norm=colors.PowerNorm(gamma=init_gamma_gx))

    ax_pl.set_ylabel('Energy (eV)')
    ax_gx.set_ylabel(r'$q$ ($\AA^{-1}$)')
    ax_gx.set_xlabel('Time (s)')

    pl_tick_locations = np.linspace(pl_energy.min(), pl_energy.max(), 4)
    gx_tick_locations = np.linspace(giwaxs_q.min(), giwaxs_q.max(), 4)

    ax_pl.set_yticks(pl_tick_locations)
    ax_pl.set_yticklabels([f"{val:.1f}" for val in pl_tick_locations])
    ax_gx.set_yticks(gx_tick_locations)
    ax_gx.set_yticklabels([f"{val:.1f}" for val in gx_tick_locations])

    ax_pl.set_title('PL Plot - Shift+Click 1: Feature Start | Shift+Click 2: Target Destination', fontsize=10)
    ax_gx.set_title('GIWAXS Plot - Click 1: Feature Start | Click 2: Target Destination | Right-Click: Reset All', fontsize=9)

    click_coords = []
    line_start, line_end = None, None

    def on_click(event):
        nonlocal click_coords, line_start, line_end, giwaxs_time, giwaxs_matrix, pl_time, pl_matrix
        if event.inaxes not in [ax_pl, ax_gx]: return

        if event.button == 3: # Reset
            if line_start is not None: line_start.remove(); line_start = None
            if line_end is not None: line_end.remove(); line_end = None
            click_coords.clear()
            ax_pl.set_title('Cleared! Shift+Click 1 to align PL timeline')
            ax_gx.set_title('Cleared! Left-Click 1 to align GIWAXS timeline')
            fig.canvas.draw_idle()
            return

        if event.button == 1: # Select
            is_pl_mode = event.key == 'shift'
            target_axes = ax_pl if is_pl_mode else ax_gx

            if len(click_coords) == 0:
                click_coords.append(event.xdata)
                line_start = target_axes.axvline(x=event.xdata, color='red', linestyle='--', linewidth=1.5)
                target_axes.set_title('Click 2: Select destination alignment mark')
                fig.canvas.draw_idle()
            elif len(click_coords) == 1:
                click_coords.append(event.xdata)
                line_end = target_axes.axvline(x=event.xdata, color='green', linestyle='--', linewidth=1.5)

                time_shift_delta = click_coords[1] - click_coords[0]

                if is_pl_mode:
                    current_pl_shift[0] += time_shift_delta
                    pl_time = pl_time_raw + current_pl_shift[0]
                    mask = (pl_time >= 0) & (pl_time <= MAX_DISPLAY_TIME)
                    pl_matrix = pl_matrix_raw[:, mask]
                    img_pl.set_data(pl_matrix)
                    img_pl.set_norm(colors.PowerNorm(gamma=slider_pl.val, vmin=pl_matrix.min(), vmax=pl_matrix.max()))
                    ax_pl.set_title(f'PL Shift Adjusted: {current_pl_shift[0]:+.1f}s')
                else:
                    current_giwaxs_shift[0] += time_shift_delta
                    giwaxs_time = giwaxs_time_raw + current_giwaxs_shift[0]
                    mask = (giwaxs_time >= 0) & (giwaxs_time <= MAX_DISPLAY_TIME)
                    giwaxs_matrix = giwaxs_matrix_raw[:, mask]
                    img_gx.set_data(giwaxs_matrix)
                    img_gx.set_extent([giwaxs_time[mask].min(), giwaxs_time[mask].max(), giwaxs_q.min(), giwaxs_q.max()])
                    img_gx.set_norm(colors.PowerNorm(gamma=slider_gx.val, vmin=giwaxs_matrix.min(), vmax=giwaxs_matrix.max()))
                    ax_gx.set_title(f'GIWAXS Shift Adjusted: {current_giwaxs_shift[0]:+.1f}s')

                if line_start is not None: line_start.remove(); line_start = None
                if line_end is not None: line_end.remove(); line_end = None
                click_coords.clear()
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    # --- GUI CONTROL BLOCKS ---
    ax_slider_pl = plt.axes([0.35, 0.09, 0.50, 0.02])
    slider_pl    = Slider(ax=ax_slider_pl, label='PL Gamma ', valmin=0.05, valmax=2.0, valinit=init_gamma_pl, valfmt='%.2f', color='gold')
    ax_slider_gx = plt.axes([0.35, 0.04, 0.50, 0.02])
    slider_gx    = Slider(ax=ax_slider_gx, label='GIWAXS Gamma ', valmin=0.05, valmax=2.0, valinit=init_gamma_gx, valfmt='%.2f', color='orangered')

    ax_radio_pl  = plt.axes([0.02, 0.55, 0.12, 0.20])
    radio_pl     = RadioButtons(ax_radio_pl, CMAP_OPTIONS, active=0, activecolor='gold')
    ax_radio_pl.set_title('PL Color', fontsize=9)
    ax_radio_gx  = plt.axes([0.02, 0.25, 0.12, 0.20])
    radio_gx     = RadioButtons(ax_radio_gx, CMAP_OPTIONS, active=1, activecolor='orangered')
    ax_radio_gx.set_title('GIWAXS Color', fontsize=9)

    def update_plots(val):
        img_pl.set_norm(colors.PowerNorm(gamma=slider_pl.val, vmin=pl_matrix.min(), vmax=pl_matrix.max()))
        img_pl.set_cmap(radio_pl.value_selected)
        img_gx.set_cmap(radio_gx.value_selected)
        fig.canvas.draw_idle()

    slider_pl.on_changed(update_plots)
    slider_gx.on_changed(update_plots)
    radio_pl.on_clicked(update_plots)
    radio_gx.on_clicked(update_plots)

    # =============================================================================
    # EXPORT ENGINE
    # =============================================================================
    ax_btn_export = plt.axes([0.02, 0.05, 0.15, 0.05])
    btn_export    = Button(ax_btn_export, 'Export Pub Data', color='lightgreen', hovercolor='green')

    def run_export_pipeline(event):
        ax_pl.set_title("")
        ax_gx.set_title("")

        ax_slider_pl.set_visible(False)
        ax_slider_gx.set_visible(False)
        ax_radio_pl.set_visible(False)
        ax_radio_gx.set_visible(False)
        ax_btn_export.set_visible(False)

        plt.subplots_adjust(bottom=0.12, left=0.15, hspace=0.1)
        fig.canvas.draw()

        print("\n[+] Select location to save publication export pack...")
        export_base = filedialog.asksaveasfilename(title="Save Publication Export Base Name", filetypes=[("All Files", "*.*")])
        if not export_base:
            ax_slider_pl.set_visible(True); ax_slider_gx.set_visible(True)
            ax_radio_pl.set_visible(True); ax_radio_gx.set_visible(True); ax_btn_export.set_visible(True)
            plt.subplots_adjust(bottom=0.25, left=0.25, hspace=0.15)
            fig.canvas.draw_idle()
            return

        fig.savefig(f"{export_base}_figure.png", dpi=600, bbox_inches='tight')
        print(f"[+] Saved Print-Ready Dashboard: {export_base}_figure.png")

        final_pl_time = pl_time_raw + current_pl_shift[0]
        pl_mask_final = (final_pl_time >= 0) & (final_pl_time <= MAX_DISPLAY_TIME)
        df_pl = pd.DataFrame(data=pl_matrix_raw[:, pl_mask_final], index=pl_energy, columns=final_pl_time[pl_mask_final])
        df_pl.to_csv(f"{export_base}_PL_intensity_matrix.csv")

        final_giwaxs_time = giwaxs_time_raw + current_giwaxs_shift[0]
        giwaxs_mask_final = (final_giwaxs_time >= 0) & (final_giwaxs_time <= MAX_DISPLAY_TIME)
        df_gx = pd.DataFrame(data=giwaxs_matrix_raw[:, giwaxs_mask_final], index=giwaxs_q, columns=final_giwaxs_time[giwaxs_mask_final])
        df_gx.to_csv(f"{export_base}_GIWAXS_intensity_matrix.csv")
        print("[+] Shifted data matrices successfully exported to CSV.")

        with open(f"{export_base}_experiment_manifest.txt", "w", encoding="utf-8") as tracker:
            tracker.write("====================================================\n")
            tracker.write("IN-SITU RUN METADATA AUDIT TRAILS MANIFEST\n")
            tracker.write("====================================================\n")
            tracker.write(f"Active Calibration File (.PONI):     {poni_path}\n")
            tracker.write(f"PL Raw Folder Location Source:      {pl_path}\n")
            tracker.write(f"GIWAXS Raw Folder Location Source:  {giwaxs_raw_path}\n")
            tracker.write(f"PL Calculated Axis Time Shift:       {current_pl_shift[0]:.3f} seconds\n")
            tracker.write(f"GIWAXS Calculated Axis Time Shift:   {current_giwaxs_shift[0]:.3f} seconds\n")
            tracker.write("====================================================\n")
        print(f"[+] Manifest Audit Trail Saved: {export_base}_experiment_manifest.txt")

        ax_slider_pl.set_visible(True); ax_slider_gx.set_visible(True)
        ax_radio_pl.set_visible(True); ax_radio_gx.set_visible(True); ax_btn_export.set_visible(True)
        plt.subplots_adjust(bottom=0.25, left=0.25, hspace=0.15)
        ax_pl.set_title('PL Plot - Shift+Click 1: Feature Start | Shift+Click 2: Target Destination', fontsize=10)
        ax_gx.set_title('GIWAXS Plot - Click 1: Feature Start | Click 2: Target Destination | Right-Click: Reset All', fontsize=9)
        fig.canvas.draw_idle()

    btn_export.on_clicked(run_export_pipeline)
    plt.show(block=True)

if __name__ == "__main__":
    main()
