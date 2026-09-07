# -*- coding: utf-8 -*-
"""
High-Performance In-situ Production Studio - Publication Matcher
Optimized with automated C-parsers, 10-core parallel scaling, local binary GIWAXS caching,
decoupled raw PL pcolormesh smoothing, publication-matched aligned layout boundaries,
dynamic gamma-corrected colorbar markers (exactly 4 ticks, formatted to 1 decimal place),
clean inward ticks, and cleared labels.
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
import warnings
from concurrent.futures import ProcessPoolExecutor

CMAP_OPTIONS = ['plasma', 'inferno', 'viridis', 'magma', 'jet', 'gray']

# =============================================================================
# HARDCODED DATA PROFILES & GLOBAL LOG SHARING
# =============================================================================
ACTIVE_SAMPLE_KEY = 'control'
SHARED_LOG_FILE = r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_ABA_S1_30_tube_5min 003294.txt"

DATA_PROFILES = {
    'control': {
        'pl_folder': r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_Control_tube_S1_36_5min\MAPI_Control_tube_S1_36_5min 003225 Spectrums",
        'giwaxs_folder': r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_Control_tube_S1_36_5min",
        'log_file': SHARED_LOG_FILE
    },
    '3APA': {
        'pl_folder': r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_APA_S1_30_tube_5min\MAPI_1pct_APA_S1_30_tube_5min 003292 Spectrums",
        'giwaxs_folder': r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_APA_S1_30_tube_5min\GIWAXS",
        'log_file': SHARED_LOG_FILE
    },
    '4ABA': {
        'pl_folder': r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_ABA_S1_30_tube_5min\MAPI_1pct_ABA_S1_30_tube_5min 003294 Spectrums",
        'giwaxs_folder': r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_ABA_S1_30_tube_5min",
        'log_file': SHARED_LOG_FILE
    },
    '5AVA': {
        'pl_folder': r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_AVA_S1_18_5min -done\MAPI_1pct_AVA_S2_30_tube_5min 003257 Spectrums",
        'giwaxs_folder': r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_AVA_S1_18_30min - good PL",
        'log_file': SHARED_LOG_FILE
    },
    '7AHA': {
        'pl_folder': r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_AHA_S1_30_tube_5min\MAPI_1pct_AHA_S1_30_tube_5min 003296 Spectrums",
        'giwaxs_folder': r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_AHA_S1_30_tube_5min",
        'log_file': SHARED_LOG_FILE
    }
}

HARDCODED_PONI = r"C:\Users\matth\OneDrive\Desktop\Manuscript data\ITO.poni"

ACTIVE_PROFILE = DATA_PROFILES[ACTIVE_SAMPLE_KEY]
HARDCODED_LOG_FILE = ACTIVE_PROFILE['log_file']
HARDCODED_PL_FOLDER = ACTIVE_PROFILE['pl_folder']
HARDCODED_GIWAXS_FOLDER = ACTIVE_PROFILE['giwaxs_folder']
GIWAXS_FIXED_SHIFT = 0


def load_in_situ_pl(folder_path, max_time=275):
    if not folder_path or not os.path.exists(folder_path):
        print(f"[-] PL path invalid or not found: {folder_path}")
        return None, None, None, ""
    current_folder = os.path.basename(os.path.normpath(folder_path))
    parent_folder  = os.path.basename(os.path.dirname(os.path.normpath(folder_path)))
    label_header   = f"{parent_folder}_{current_folder}"

    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not files:
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
    warnings.filterwarnings("ignore", category=UserWarning, module="glymur")
    full_path, poni_file_path = args
    try:
        import fabio
        import pyFAI
        file_timestamp = os.path.getmtime(full_path)
        ai = pyFAI.load(poni_file_path)
        img_frame = fabio.open(full_path).data
        res = ai.integrate1d_ng(img_frame, 500, unit="q_A^-1")
        return res.radial, res.intensity, file_timestamp
    except Exception as e:
        return None, None, None


def process_parallel_giwaxs(folder_path, poni_file_path):
    if not folder_path or not poni_file_path:
        return None, None, None

    cache_file = os.path.join(folder_path, "giwaxs_integrated_cache.npz")
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
        except Exception:
            pass

    if not os.path.exists(folder_path):
        return None, None, None

    valid_exts = ('.tif', '.tiff', '.edf', '.cbf')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    if not files:
        return None, None, None

    files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', var)])
    tasks = [(os.path.join(folder_path, f), poni_file_path) for f in files]

    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(_integrate_single_frame, tasks))

    processed_dataset = []
    q_axis = None

    for rad, inten, file_timestamp in results:
        if rad is not None:
            if q_axis is None:
                q_axis = rad
            processed_dataset.append((file_timestamp, inten))

    if not processed_dataset:
        return None, None, None

    processed_dataset.sort(key=lambda item: item[0])
    sorted_timestamps = np.array([item[0] for item in processed_dataset])
    giwaxs_matrix = np.array([item[1] for item in processed_dataset]).T

    base_time = sorted_timestamps[0]
    time_axis = sorted_timestamps - base_time + GIWAXS_FIXED_SHIFT

    try:
        np.savez_compressed(cache_file, time_axis=time_axis, q_axis=q_axis, giwaxs_matrix=giwaxs_matrix)
    except Exception:
        pass

    return time_axis, q_axis, giwaxs_matrix


def load_spincoater_log(file_path):
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        skip_rows = None
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                clean_line = line.strip().lower()
                if "time of day" in clean_line and "time (s)" in clean_line:
                    skip_rows = i
                    break

        if skip_rows is None:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if line.strip() == "DATA":
                        skip_rows = i + 1
                        break

        if skip_rows is None:
            return None

        df = pd.read_csv(file_path, skiprows=skip_rows, sep='\t', engine='python', on_bad_lines='skip')
        df.columns = [str(c).strip().lower() for c in df.columns]

        time_col = next((c for c in df.columns if 'time (s)' in c or c == 'time'), None)
        rpm_col = next((c for c in df.columns if 'spin motor' in c or 'motor' in c or 'rpm' in c), None)
        temp_col = next((c for c in df.columns if 'pyrometer' in c or 'pyro' in c or 'temp' in c), None)

        if time_col is None:
            time_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        clean_df = pd.DataFrame()
        clean_df['time'] = pd.to_numeric(df[time_col], errors='coerce')
        clean_df['rpm'] = pd.to_numeric(df[rpm_col], errors='coerce') if rpm_col else np.zeros(len(df))
        clean_df['temp'] = pd.to_numeric(df[temp_col], errors='coerce') if temp_col else np.zeros(len(df))

        return clean_df.dropna(subset=['time'])
    except Exception:
        return None


def main():
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15
    })

    MAX_DISPLAY_TIME  = 275.0
    GRID_POINTS = 1500

    pl_time_raw, pl_energy, pl_matrix_raw, pl_label = load_in_situ_pl(HARDCODED_PL_FOLDER, max_time=MAX_DISPLAY_TIME)
    if pl_matrix_raw is None: return

    if np.median(pl_energy) > 100: pl_energy = 1239.84193 / pl_energy
    pl_sort   = np.argsort(pl_energy)
    pl_energy = pl_energy[pl_sort]
    pl_matrix_raw = pl_matrix_raw[pl_sort, :]

    pl_mask   = (pl_energy >= 1.20) & (pl_energy <= 2.00)
    pl_energy = pl_energy[pl_mask]
    pl_matrix_raw = pl_matrix_raw[pl_mask, :]

    giwaxs_time_raw, giwaxs_q, giwaxs_matrix_raw = process_parallel_giwaxs(HARDCODED_GIWAXS_FOLDER, HARDCODED_PONI)
    if giwaxs_matrix_raw is None: return

    giwaxs_noise_floor = np.percentile(giwaxs_matrix_raw, 15)
    giwaxs_matrix_raw[giwaxs_matrix_raw < giwaxs_noise_floor] = giwaxs_noise_floor

    gx_sort       = np.argsort(giwaxs_q)
    giwaxs_q      = giwaxs_q[gx_sort]
    giwaxs_matrix_raw = giwaxs_matrix_raw[gx_sort, :]

    giwaxs_mask   = (giwaxs_q >= 0.5) & (giwaxs_q <= 2.5)
    giwaxs_q      = giwaxs_q[giwaxs_mask]
    giwaxs_matrix_raw = giwaxs_matrix_raw[giwaxs_mask, :]

    log_df = load_spincoater_log(HARDCODED_LOG_FILE)
    current_pl_shift = [0.0]
    current_giwaxs_shift = [0.0]
    uniform_timeline = np.linspace(0.0, MAX_DISPLAY_TIME, GRID_POINTS)

    def resample_matrix(time_raw, matrix_raw, shift_val):
        shifted_time = time_raw + shift_val
        resampled = np.zeros((matrix_raw.shape[0], GRID_POINTS))
        start_frame, end_frame = 15, matrix_raw.shape[1] - 3
        active_matrix_subset = matrix_raw[:, start_frame:end_frame]
        active_time_subset = shifted_time[start_frame:end_frame]

        for i in range(matrix_raw.shape[0]):
            resampled[i, :] = np.interp(uniform_timeline, active_time_subset, active_matrix_subset[i, :])
        return resampled

    giwaxs_matrix_grid = resample_matrix(giwaxs_time_raw, giwaxs_matrix_raw, current_giwaxs_shift[0])

    fig, (ax_pl, ax_gx, ax_tool) = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    plt.subplots_adjust(bottom=0.22, left=0.20, right=0.80, hspace=0.22)

    ax_pl.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax_gx.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax_tool.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    extent_gx = [0.0, MAX_DISPLAY_TIME, 0.50, 2.50]
    init_gamma_pl, init_gamma_gx = 0.25, 0.25

    pl_matrix_gamma = np.power(pl_matrix_raw, init_gamma_pl)
    giwaxs_matrix_gamma = np.power(giwaxs_matrix_grid, init_gamma_gx)

    pl_vmin, pl_vmax = np.percentile(pl_matrix_gamma, 1), np.percentile(pl_matrix_gamma, 99.9)
    gx_vmin, gx_vmax = np.percentile(giwaxs_matrix_gamma, 1), np.percentile(giwaxs_matrix_gamma, 99.9)

    img_pl = ax_pl.pcolormesh(pl_time_raw + current_pl_shift[0], pl_energy, pl_matrix_gamma,
                              cmap='plasma', shading='gouraud',
                              norm=colors.Normalize(vmin=pl_vmin, vmax=pl_vmax))

    img_gx = ax_gx.imshow(giwaxs_matrix_gamma, aspect='auto', cmap='inferno', origin='lower',
                           extent=extent_gx, norm=colors.Normalize(vmin=gx_vmin, vmax=gx_vmax),
                           interpolation='nearest')

    # =============================================================================
    # INITIALIZE INTENSITY BARS SET TO EXACTLY 4 MARKERS WITH 1 DECIMAL PLACE
    # =============================================================================
    cbar_pl = fig.colorbar(img_pl, ax=ax_pl, fraction=0.046, pad=0.02)
    cbar_pl.set_label('', rotation=90, labelpad=20)
    pl_ticks = np.linspace(pl_vmin, pl_vmax, 4)
    cbar_pl.set_ticks(pl_ticks, labels=[f"{x:.1f}" for x in pl_ticks]) # 1 decimal formatting
    cbar_pl.ax.tick_params(direction='in')

    cbar_gx = fig.colorbar(img_gx, ax=ax_gx, fraction=0.046, pad=0.02)
    cbar_gx.set_label('', rotation=90, labelpad=20)
    gx_ticks = np.linspace(gx_vmin, gx_vmax, 4)
    cbar_gx.set_ticks(gx_ticks, labels=[f"{x:.1f}" for x in gx_ticks]) # 1 decimal formatting
    cbar_gx.ax.tick_params(direction='in')

    if log_df is not None:
        ax_tool_temp = ax_tool.twinx()
        ax_tool_temp.tick_params(axis='both', which='both', direction='in')

        ax_tool.plot(log_df['time'], log_df['rpm'], color='blue', linewidth=2.0, label='Spin Speed')
        ax_tool_temp.plot(log_df['time'], log_df['temp'], color='crimson', linewidth=1.75, label='Substrate Temp')

        ax_tool.set_ylabel('Spin Speed (RPM)', color='blue', fontweight='bold')
        ax_tool.tick_params(axis='y', labelcolor='blue')
        ax_tool.spines['left'].set_color('blue')
        ax_tool.spines['left'].set_linewidth(2.0)
        ax_tool.set_yticks([0, 2000])
        ax_tool.set_ylim(-50, 2200)


        ax_tool_temp.set_ylabel('Temperature (°C)', color='crimson', fontweight='bold')
        ax_tool_temp.tick_params(axis='y', labelcolor='crimson')
        ax_tool_temp.spines['right'].set_color('crimson')
        ax_tool_temp.spines['right'].set_linewidth(2.0)
        ax_tool_temp.set_xlim(0.0, MAX_DISPLAY_TIME)

    ax_pl.set_ylabel('Energy (eV)')
    ax_gx.set_ylabel(r'$q$ ($\AA^{-1}$)')
    ax_tool.set_xlabel('Time (s)')

    pl_tick_locations = [1.20, 1.60, 2.00]
    ax_pl.set_yticks(pl_tick_locations)
    ax_pl.set_yticklabels([f"{val:.2f}" for val in pl_tick_locations])
    ax_pl.set_ylim(1.20, 2.00)

    gx_tick_locations = [0.50, 1.00, 1.50, 2.00, 2.50]
    ax_gx.set_yticks(gx_tick_locations)
    ax_gx.set_yticklabels([f"{val:.2f}" for val in gx_tick_locations])
    ax_gx.set_ylim(0.50, 2.50)

    fig.canvas.draw()
    pos_pl = ax_pl.get_position()
    pos_tool = ax_tool.get_position()
    ax_tool.set_position([pos_pl.x0, pos_tool.y0, pos_pl.width, pos_tool.height])

    click_coords = []
    line_start, line_end = None, None

    def on_click(event):
        nonlocal click_coords, line_start, line_end, giwaxs_matrix_grid, img_pl
        if event.inaxes not in [ax_pl, ax_gx, ax_tool]: return
        if event.button == 3:
            if line_start is not None: line_start.remove(); line_start = None
            if line_end is not None: line_end.remove(); line_end = None
            click_coords.clear()
            fig.canvas.draw_idle()
            return
        if event.button == 1:
            is_pl_mode = event.key == 'shift'
            target_axes = ax_pl if is_pl_mode else ax_gx
            if len(click_coords) == 0:
                click_coords.append(event.xdata)
                line_start = target_axes.axvline(x=event.xdata, color='red', linestyle='--', linewidth=1.5)
                fig.canvas.draw_idle()
            elif len(click_coords) == 1:
                click_coords.append(event.xdata)
                line_end = target_axes.axvline(x=event.xdata, color='green', linestyle='--', linewidth=1.5)
                time_shift_delta = click_coords[1] - click_coords[0]

                if is_pl_mode:
                    current_pl_shift[0] += time_shift_delta
                    img_pl.remove()
                    active_pl_gamma = np.power(pl_matrix_raw, slider_pl.val)
                    dyn_pl_vmin, dyn_pl_vmax = np.percentile(active_pl_gamma, 1), np.percentile(active_pl_gamma, 99.9)
                    img_pl = ax_pl.pcolormesh(pl_time_raw + current_pl_shift[0], pl_energy, active_pl_gamma,
                                              cmap=radio_pl.value_selected, shading='gouraud',
                                              norm=colors.Normalize(vmin=dyn_pl_vmin, vmax=dyn_pl_vmax))
                    new_pl_ticks = np.linspace(dyn_pl_vmin, dyn_pl_vmax, 4)
                    cbar_pl.set_ticks(new_pl_ticks, labels=[f"{x:.1f}" for x in new_pl_ticks])
                else:
                    current_giwaxs_shift[0] += time_shift_delta
                    giwaxs_matrix_grid = resample_matrix(giwaxs_time_raw, giwaxs_matrix_raw, current_giwaxs_shift[0])
                    active_gx_gamma = np.power(giwaxs_matrix_grid, slider_gx.val)
                    dyn_gx_vmin, dyn_gx_vmax = np.percentile(active_gx_gamma, 1), np.percentile(active_gx_gamma, 99.9)
                    img_gx.set_data(active_gx_gamma)
                    img_gx.set_norm(colors.Normalize(vmin=dyn_gx_vmin, vmax=dyn_gx_vmax))
                    new_gx_ticks = np.linspace(dyn_gx_vmin, dyn_gx_vmax, 4)
                    cbar_gx.set_ticks(new_gx_ticks, labels=[f"{x:.1f}" for x in new_gx_ticks])

                if line_start is not None: line_start.remove(); line_start = None
                if line_end is not None: line_end.remove(); line_end = None
                click_coords.clear()
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    ax_slider_pl = plt.axes([0.35, 0.08, 0.50, 0.02])
    slider_pl    = Slider(ax=ax_slider_pl, label='PL Gamma ', valmin=0.05, valmax=2.0, valinit=init_gamma_pl, valfmt='%.2f', color='gold')
    ax_slider_gx = plt.axes([0.35, 0.03, 0.50, 0.02])
    slider_gx    = Slider(ax=ax_slider_gx, label='GIWAXS Gamma ', valmin=0.05, valmax=2.0, valinit=init_gamma_gx, valfmt='%.2f', color='orangered')

    ax_radio_pl  = plt.axes([0.02, 0.60, 0.12, 0.18])
    radio_pl     = RadioButtons(ax_radio_pl, CMAP_OPTIONS, active=0, activecolor='gold')
    ax_radio_gx  = plt.axes([0.02, 0.35, 0.12, 0.18])
    radio_gx     = RadioButtons(ax_radio_gx, CMAP_OPTIONS, active=1, activecolor='orangered')

    # =============================================================================
    # REAL-TIME DYNAMIC MATH UPDATES RESTRICTED TO EXACTLY 4 TICK MARKERS AT 1 DECIMAL
    # =============================================================================
    def update_plots(val):
        updated_pl_gamma = np.power(pl_matrix_raw, slider_pl.val)
        dyn_pl_vmin, dyn_pl_vmax = np.percentile(updated_pl_gamma, 1), np.percentile(updated_pl_gamma, 99.9)
        img_pl.set_array(updated_pl_gamma.ravel())
        img_pl.set_norm(colors.Normalize(vmin=dyn_pl_vmin, vmax=dyn_pl_vmax))
        img_pl.set_cmap(radio_pl.value_selected)
        cbar_pl.set_label('', rotation=90, labelpad=20)
        new_pl_ticks = np.linspace(dyn_pl_vmin, dyn_pl_vmax, 4)
        cbar_pl.set_ticks(new_pl_ticks, labels=[f"{x:.1f}" for x in new_pl_ticks])

        updated_gx_gamma = np.power(giwaxs_matrix_grid, slider_gx.val)
        dyn_gx_vmin, dyn_gx_vmax = np.percentile(updated_gx_gamma, 1), np.percentile(updated_gx_gamma, 99.9)
        img_gx.set_data(updated_gx_gamma)
        img_gx.set_norm(colors.Normalize(vmin=dyn_gx_vmin, vmax=dyn_gx_vmax))
        img_gx.set_cmap(radio_gx.value_selected)
        cbar_gx.set_label('', rotation=90, labelpad=20)
        new_gx_ticks = np.linspace(dyn_gx_vmin, dyn_gx_vmax, 4)
        cbar_gx.set_ticks(new_gx_ticks, labels=[f"{x:.1f}" for x in new_gx_ticks])
        fig.canvas.draw_idle()

    slider_pl.on_changed(update_plots)
    slider_gx.on_changed(update_plots)
    radio_pl.on_clicked(update_plots)
    radio_gx.on_clicked(update_plots)

    ax_btn_export = plt.axes([0.02, 0.05, 0.15, 0.05])
    btn_export    = Button(ax_btn_export, 'Export Pub Data', color='lightgreen', hovercolor='green')

    def run_export_pipeline(event):
        ax_slider_pl.set_visible(False); ax_slider_gx.set_visible(False)
        ax_radio_pl.set_visible(False); ax_radio_gx.set_visible(False); ax_btn_export.set_visible(False)
        plt.subplots_adjust(bottom=0.10, left=0.15, right=0.85, hspace=0.12)
        fig.canvas.draw()

        p_pl = ax_pl.get_position()
        p_tl = ax_tool.get_position()
        ax_tool.set_position([p_pl.x0, p_tl.y0, p_pl.width, p_tl.height])
        fig.canvas.draw()

        root = tk.Tk()
        root.withdraw()
        export_base = filedialog.asksaveasfilename(title="Save Publication Export Base Name")
        if not export_base:
            ax_slider_pl.set_visible(True); ax_slider_gx.set_visible(True)
            ax_radio_pl.set_visible(True); ax_radio_gx.set_visible(True); ax_btn_export.set_visible(True)
            plt.subplots_adjust(bottom=0.22, left=0.20, right=0.80, hspace=0.22)
            fig.canvas.draw_idle()
            return

        fig.savefig(f"{export_base}_figure.png", dpi=600, bbox_inches='tight')

        pl_export_matrix = np.power(pl_matrix_raw, slider_pl.val)
        giwaxs_export_matrix = np.power(giwaxs_matrix_grid, slider_gx.val)

        pd.DataFrame(data=pl_export_matrix, index=pl_energy, columns=pl_time_raw + current_pl_shift[0]).to_csv(f"{export_base}_Aligned_PL_matrix.csv")
        pd.DataFrame(data=giwaxs_export_matrix, index=giwaxs_q, columns=uniform_timeline).to_csv(f"{export_base}_Aligned_GIWAXS_matrix.csv")

        ax_slider_pl.set_visible(True); ax_slider_gx.set_visible(True)
        ax_radio_pl.set_visible(True); ax_radio_gx.set_visible(True); ax_btn_export.set_visible(True)
        plt.subplots_adjust(bottom=0.22, left=0.20, right=0.80, hspace=0.22)
        fig.canvas.draw_idle()

    btn_export.on_clicked(run_export_pipeline)
    plt.show(block=True)


if __name__ == "__main__":
    main()
