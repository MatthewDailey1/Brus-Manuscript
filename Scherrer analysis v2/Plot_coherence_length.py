import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# import tkinter as tk
# from tkinter import filedialog, ttk
# import tkinter.messagebox as messagebox
from scipy.signal import savgol_filter

# import matplotlib
# matplotlib.use('TkAgg')
plt.style.use('seaborn-v0_8-whitegrid')

# =========================================================================
# EXPERIMENTAL CONFIGURATION & EXACT PATHS
# =========================================================================
SAMPLE_ORDER = ['control', '3-APA', '4-ABA', '5-AVA', '7-AHA']

SAMPLE_STYLES = {
    'control': {'color': 'purple',       'marker': 'o',  'label': 'Control'},
    '3-APA':   {'color': 'blue',       'marker': 'o',  'label': '3-APA'},
    '4-ABA':   {'color': 'orange',   'marker': 's',  'label': '4-ABA'},
    '5-AVA':   {'color': 'green', 'marker': '^',  'label': '5-AVA'},
    '7-AHA':   {'color': 'crimson',     'marker': 'D',  'label': '7-AHA'}
}

# TODO: Replace these placeholder strings with your exact local file paths
FILE_PATHS = {
    'control': r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_Control_tube_S1_36_5min\processed_data\redo on this - better data\scherrer_kinetics_output-Control.csv",
    '5-AVA':   r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_1pct_AVA_S1_18_5min -done\processed_data\scherrer_v2\scherrer_kinetics_output_AVA.csv",
    '4-ABA':   r"G:\LBL_data\printz_Apr2024\MAPI_sean\MAPI_1pct_ABA_S1_30_tube_5min\processed_data\scherrer_kinetics_automated_output.csv",
    '3-APA':   r"G:\LBL_data\printz_Apr2024\MAPI_sean\MAPI_1pct_APA_S1_30_tube_5min\processed_data\scherrer_kinetics_automated_output.csv",
    '7-AHA':   r"G:\LBL_data\printz_Apr2024\MAPI_sean\MAPI_1pct_AHA_S1_30_tube_5min\processed_data\scherrer_kinetics_automated_output.csv"
}
# =========================================================================
# AUTOMATED FILE LOADING
# =========================================================================
found_datasets = {}

print("Loading datasets from explicit file paths...")

for sample_key, path_str in FILE_PATHS.items():
    if not path_str:
        continue

    path_obj = Path(path_str)

    if not path_obj.exists():
        print(f"[!] Warning: Path defined for [{sample_key}] does not exist: {path_str}")
        continue

    try:
        df = pd.read_csv(path_obj, engine='python')

        size_col = [col for col in df.columns if 'size' in col.lower() or 'scherrer' in col.lower()]
        time_col = [col for col in df.columns if 'time' in col.lower() or 'frame' in col.lower()]

        if size_col and time_col:
            df = df.rename(columns={time_col[0]: 'Time_Value', size_col[0]: 'Scherrer_Domain_Size_nm'})
            found_datasets[sample_key] = df
            print(f"--> Successfully loaded [{sample_key}] from '{path_obj.name}'")
        else:
            print(f"[!] Error: File '{path_obj.name}' read successfully but missing recognizable Time or Size headers.")
    except Exception as e:
        print(f"[!] Critical Error reading file for [{sample_key}]: {e}")

if not found_datasets:
    sys.exit("\n[Exit] No valid datasets were successfully loaded. Plotting cancelled.")

# =========================================================================
# CORE KINETICS VISUALIZATION
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 6))

legend_handles = []

for sample_key in SAMPLE_ORDER:
    if sample_key in found_datasets:
        df = found_datasets[sample_key]
        style = SAMPLE_STYLES[sample_key]

        time_vals = df['Time_Value'].values
        size_nm = df['Scherrer_Domain_Size_nm'].values.copy()

        # SMOOTHING ENGINE: Apply Savitzky-Golay filtering only to the noisy AVAI dataset
        if sample_key == 'AVAI':
            window_len = min(9, len(size_nm))
            if window_len % 2 == 0:
                window_len -= 1

            if window_len >= 5:
                size_nm = savgol_filter(size_nm, window_length=window_len, polyorder=2)

        lines = ax.plot(
            time_vals,
            size_nm,
            linestyle='-',
            linewidth=2.5,
            marker=style['marker'],
            color=style['color'],
            markersize=5,
            markeredgecolor='white',
            markeredgewidth=0.5
        )

        line = lines[0]
        line.set_label(style['label'])
        legend_handles.append(line)

# Publication formatting adjustments
ax.set_xlabel("Time (s)", fontweight='bold', fontsize=12)
ax.set_ylabel(r"Coherence Length $L_c$ (nm)", fontweight='bold', fontsize=12)
ax.set_title("Coherence Length from GIWAXS", fontsize=13, fontweight='bold', pad=15)

# FIXED: Removed the 10.0 scaling factor so y-lim reflects nm scale properly
all_lengths = [val for df in found_datasets.values() for val in df['Scherrer_Domain_Size_nm'].values]
if all_lengths:
    ax.set_ylim(0, max(all_lengths) * 1.15)

ax.legend(handles=legend_handles, loc='best', frameon=True, shadow=True, facecolor='white', edgecolor='gainsboro', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
print("\n[DISPLAY] Rendering final clean multi-sample compilation figure window...")
plt.show()
