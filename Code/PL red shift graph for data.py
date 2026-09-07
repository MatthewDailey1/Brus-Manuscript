# -*- coding: utf-8 -*-
"""
Multi-Dataset Time-Resolved PL Comparison (Single Window: 0 to 85s)
=============================================================================
1. Iterates through multiple specified dataset directories.
2. Reads actual OS modification times to calculate real elapsed time.
3. Filters data strictly to the 0-85s window.
4. Downsamples to 30 files for high-resolution trace comparison.
5. Dynamically skips text headers, converts to eV, and smooths the data.
6. Auto-scales the Y-axis and generates a clean, single-panel plot per dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Configure global matplotlib parameters for clean Arial formatting
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# =============================================================================
# MULTI-DATASET CONFIGURATION
# =============================================================================
DATASETS = [
    {
        "title": "MAPI Control Tube",
        "path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_Control_tube_S1_36_5min\MAPI_Control_tube_S1_36_5min 003225 Spectrums"
    },
    {
        "title": "MAPI 1% 3-APA Tube",
        "path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_APA_S1_30_tube_5min\MAPI_1pct_APA_S1_30_tube_5min 003292 Spectrums"
    },
    {
        "title": "MAPI 1% 4-ABA Tube",
        "path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_ABA_S1_30_tube_5min\MAPI_1pct_ABA_S1_30_tube_5min 003294 Spectrums"
    },
    {
        "title": "MAPI 1% 5-AVA Tube",
        "path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_AVA_S1_18_5min -done\MAPI_1pct_AVA_S2_30_tube_5min 003257 Spectrums"
    },
    {
        "title": "MAPI 1% 7-AHA Tube",
        "path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_AHA_S1_30_tube_5min\MAPI_1pct_AHA_S1_30_tube_5min 003296 Spectrums"
    }
]

def parse_and_smooth_files(file_list):
    """Parses a list of text files, skips headers, converts to eV, and smooths."""
    parsed_signals = []
    for file_path in file_list:
        try:
            skip_lines = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if ">>>>>Begin Spectral Data<<<<<" in line:
                        skip_lines = i + 1
                        break

            df = pd.read_csv(file_path, skiprows=skip_lines, comment='#', header=None, sep=r'\s+', engine='python')

            if df.shape[1] < 2:
                df = pd.read_csv(file_path, skiprows=skip_lines, comment='#', header=None, sep=',', engine='python')

            if df.shape[1] < 2:
                continue

            x_wavelength = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            y_counts = pd.to_numeric(df.iloc[:, 1], errors='coerce')

            valid_mask = ~(x_wavelength.isna() | y_counts.isna())
            x_valid = x_wavelength[valid_mask]
            y_valid = y_counts[valid_mask]

            x_energy = 1240.0 / x_valid

            sort_indices = np.argsort(x_energy)
            x_sorted = x_energy.iloc[sort_indices]
            y_sorted = y_valid.iloc[sort_indices]

            # Apply Savitzky-Golay smoothing
            y_smoothed = savgol_filter(y_sorted, window_length=21, polyorder=3)
            parsed_signals.append((x_sorted, y_smoothed))
        except Exception as e:
            print(f"[!] Error reading file {os.path.basename(file_path)}: {e}")

    return parsed_signals

def plot_all_datasets():
    """Loops through all datasets, filters to 0-85s using real timestamps, and plots single figure."""

    max_target_time = 120
    num_signals = 30  # Number of lines per plot

    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=0, vmax=max_target_time)

    for dataset in DATASETS:
        print(f"\nProcessing: {dataset['title']}")
        target_dir = dataset["path"]

        if not os.path.exists(target_dir):
            print(f"[!] Directory not found: {target_dir}")
            continue

        raw_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.txt') and not f.startswith('._')]

        if not raw_files:
            print(f"[!] No valid data files found in: {target_dir}")
            continue

        # Extract real file modification timestamps
        file_time_data = []
        for f in raw_files:
            file_time_data.append((f, os.path.getmtime(f)))

        # Sort files chronologically
        file_time_data.sort(key=lambda x: x[1])

        # Calculate elapsed times relative to the first file (t0 = 0s)
        t0 = file_time_data[0][1]
        all_files = [item[0] for item in file_time_data]
        elapsed_times = np.array([item[1] - t0 for item in file_time_data])

        # Filter strictly to files between 0s and 85s
        idx_target = np.where(elapsed_times <= max_target_time)[0]
        files_target = [all_files[i] for i in idx_target]
        times_target = elapsed_times[idx_target]

        if not files_target:
            print(f"[!] No files found within 0-{max_target_time}s for {dataset['title']}")
            continue

        # Downsample to 30 evenly spaced files across the 0-85s window
        num_to_take = min(num_signals, len(files_target))
        ds_indices = np.linspace(0, len(files_target) - 1, num_to_take, dtype=int)
        selected_files = [files_target[i] for i in ds_indices]
        selected_times = [times_target[i] for i in ds_indices]

        print(f"✓ Selected {len(selected_files)} files across real time range: {selected_times[0]:.1f}s to {selected_times[-1]:.1f}s")

        # Parse and smooth selected data
        signals = parse_and_smooth_files(selected_files)

        # Initialize single panel figure
        fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=300)
        fig.patch.set_alpha(1.0)

        global_max_y = float('-inf')
        global_min_y = float('inf')

        # Plot traces mapped to exact real elapsed time
        for idx, (x_vals, y_vals) in enumerate(signals):
            t_val = selected_times[idx]
            ax.plot(x_vals, y_vals, color=cmap(norm(t_val)), linewidth=1.5, alpha=0.9)

            if y_vals.max() > global_max_y: global_max_y = y_vals.max()
            if y_vals.min() < global_min_y: global_min_y = y_vals.min()

        # Formatting Panel Properties
        ax.set_title(f"{dataset['title']} (0 - {max_target_time:.0f}s)", fontsize=13, fontname='Arial', pad=8, weight='bold')
        ax.set_xlabel('Energy (eV)', fontsize=13, fontname='Arial', labelpad=8)
        ax.set_ylabel('Intensity (Counts)', fontsize=13, fontname='Arial', labelpad=8)
        ax.set_xlim(1.3, 2.1)

        # Auto-scale Y limits based on data in this 85s window (with 5% buffer)
        y_range = global_max_y - global_min_y
        ax.set_ylim(global_min_y - (0.05 * y_range), global_max_y + (0.05 * y_range))

        # Tick and Spine Styling
        ax.tick_params(axis='both', direction='in', top=True, right=True, width=1.0, length=5, labelsize=11)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Arial')
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        # Add Colorbar (0 to 85s)
        plt.subplots_adjust(left=0.15, right=0.82, top=0.90, bottom=0.15)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Time (s)', fontsize=12, fontname='Arial', labelpad=8)
        cbar.ax.tick_params(labelsize=11)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Arial')
        cbar.outline.set_linewidth(1.0)

        # Save
        output_path = os.path.join(r"C:\Users\matth\Downloads", f"{dataset['title']}_0_85s.png")
        plt.savefig(output_path, dpi=300, transparent=True, facecolor='none')

    print("\nDisplaying all plots...")
    plt.show()

if __name__ == "__main__":
    plot_all_datasets()
