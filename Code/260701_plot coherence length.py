import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

# Use a clean base style
plt.style.use('seaborn-v0_8-whitegrid')

# =========================================================================
# EXPERIMENTAL CONFIGURATION & EXACT PATHS
# =========================================================================
SAMPLE_ORDER = ['control', '3-APA', '4-ABA', '5-AVA', '7-AHA']

SAMPLE_STYLES = {
    'control': {'color': 'black',       'marker': 'o',  'label': 'Control'},
    '3-APA':   {'color': 'red',         'marker': 'o',  'label': '3-APA'},
    '4-ABA':   {'color': 'green',       'marker': 's',  'label': '4-ABA'},
    '5-AVA':   {'color': 'purple',      'marker': '^',  'label': '5-AVA'},
    '7-AHA':   {'color': 'orange',      'marker': 'D',  'label': '7-AHA'}
}

FILE_PATHS = {
    'control': r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_Control_tube_S1_36_5min\processed_data\redo on this - better data\scherrer_kinetics_output-Control.csv",
    '5-AVA':   r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_1pct_AVA_S1_18_5min -done\processed_data\scherrer_v2\scherrer_kinetics_output_AVA.csv",
    '4-ABA':   r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_1pct_ABA_S1_30_tube_5min\processed_data_v2\scherrer_kinetics_automated_output.csv",
    '3-APA':   r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_1pct_APA_S1_30_tube_5min\processed_data\scherrer_kinetics_automated_output.csv",
    '7-AHA':   r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_1pct_AHA_S1_30_tube_5min\processed_data_v2\scherrer_kinetics_automated_output.csv"
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
        try:
            df = pd.read_csv(path_obj, engine='python', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(path_obj, engine='python', encoding='cp1252')

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
# CORE KINETICS VISUALIZATION (PUBLICATION QUALITY STYLE)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 6.5))
legend_handles = []

for sample_key in SAMPLE_ORDER:
    if sample_key in found_datasets:
        df = found_datasets[sample_key]
        style = SAMPLE_STYLES[sample_key]

        time_vals = df['Time_Value'].values
        size_nm = df['Scherrer_Domain_Size_nm'].values.copy()

        # Apply Savitzky-Golay filtering if applicable
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

# -----------------------------------------------------------------
# HIGHLIGHTING REGIONS (Shaded spans + Clean Uniform Text) - COMMENTED OUT
# -----------------------------------------------------------------
# r1_end, r2_end = 80, 105
# TEXT_COLOR = '#333333'  # Uniform clean dark gray text
# TEXT_Y_POS = 73.0       # Dynamically positioned near the top of the 80 nm scale

# Region I: Baseline/Incubation (50 to 80s)
# ax.axvspan(50, r1_end, color='gray', alpha=0.06)
# ax.text(65.0, TEXT_Y_POS, "Region I", color=TEXT_COLOR, fontsize=12,
#         fontweight='bold', ha='center', va='center')

# Region II: Peak Phase (80 to 105s)
# ax.axvspan(r1_end, r2_end, color='orange', alpha=0.04)
# ax.text(92.5, TEXT_Y_POS, "Region II", color=TEXT_COLOR, fontsize=12,
#         fontweight='bold', ha='center', va='center')

# Region III: Steady Plateau Phase (105 to 275s)
# ax.axvspan(r2_end, 275, color='blue', alpha=0.03)
# ax.text(190.0, TEXT_Y_POS, "Region III", color=TEXT_COLOR, fontsize=12,
#         fontweight='bold', ha='center', va='center')

# Clean subtle vertical dividers
# ax.axvline(x=r1_end, color='black', linestyle=':', linewidth=1.0, alpha=0.4)
# ax.axvline(x=r2_end, color='black', linestyle=':', linewidth=1.0, alpha=0.4)

# -----------------------------------------------------------------
# PUBLICATION FORMATTING & FRAME (BLACK BOX STYLE)
# -----------------------------------------------------------------
ax.set_xlabel("Time (s)", fontsize=13, fontweight='bold', labelpad=8)
ax.set_ylabel(r"Coherence Length $L_c$ (nm)", fontsize=13, fontweight='bold', labelpad=8)

# Limits and Ticks Step Configuration
ax.set_xlim(50, 275)
ax.set_ylim(0, 75)

ax.set_xticks(np.arange(50, 276, 50))   # Ticks every 25s from 50 to 275
ax.set_yticks(np.arange(0, 76, 25))     # Ticks every 20 nm from 0 to 80

# Remove Gridlines entirely for clean presentation
ax.grid(False)

# Enforce Classic Black Box Framework around full figure
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.5)

# Force Ticks Inwards on All 4 Borders (Bottom, Top, Left, Right)
ax.tick_params(
    direction='in',
    top=True,
    right=True,
    bottom=True,
    left=True,
    length=6,
    width=1.5,
    colors='black',
    labelsize=12
)

# Clean borderless legend placed back directly in the top right corner
ax.legend(
    handles=legend_handles,
    loc='upper right',
    frameon=False,
    fontsize=11
)

plt.tight_layout()
print("\n[DISPLAY] Rendering final clean multi-sample compilation figure window...")
plt.show()
