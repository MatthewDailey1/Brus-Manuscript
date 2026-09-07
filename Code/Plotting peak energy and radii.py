# -*- coding: utf-8 -*-
"""
Radius & Peak Energy Dual-Axis Suite ("RadiusResults_All" Sheet)
=============================================================================
1. Reads the master data spreadsheet.
2. Generates publication-ready 1:1 square dual Y-axis plots with smaller
   markers and thinner lines.
3. Saves each output directly to the user's Downloads directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Global typography and aesthetic setup
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['savefig.transparent'] = True

# =============================================================================
# DATA PIPELINE CONFIGURATION
# =============================================================================
EXCEL_PATH = r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\260717_Master_Data.xlsx"
SHEET_NAME = "RadiusResults_All"

# Samples to process individually
SAMPLES = ["control", "3-APA", "4-ABA", "5-AVA", "7-AHA"]

def load_sample_series(sample_name):
    """Loads the full series for a specific sample, retaining original alignment."""
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Could not locate spreadsheet at: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    energy_col = None
    radius_col = None
    time_col = None

    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()

        normalized_sample = sample_name.lower().replace("-", "")
        normalized_col = col_lower.replace("-", "")

        if normalized_sample in normalized_col:
            if 'energy' in col_lower or 'peak' in col_lower:
                energy_col = col_str
            elif 'radius' in col_lower or 'rad' in col_lower:
                radius_col = col_str

        # Look for a time or frame column if present in the sheet
        if 'time' in col_lower or 'frame' in col_lower or 'sec' in col_lower:
            if not time_col:
                time_col = col_str

    if not energy_col or not radius_col:
        raise ValueError(f"Could not find matching columns for sample: {sample_name}")

    sub_df = df[[col for col in [time_col, energy_col, radius_col] if col is not None]].dropna()

    if time_col and time_col in sub_df.columns:
        x_data = sub_df[time_col].reset_index(drop=True)
    else:
        x_data = pd.Series(np.arange(len(sub_df)))

    energy_data = sub_df[energy_col].reset_index(drop=True)
    radius_data = sub_df[radius_col].reset_index(drop=True)

    return x_data, energy_data, radius_data

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_sample_dual_axis(sample_name):
    """Generates a 1:1 square dual Y-axis plot and saves it to Downloads."""
    x_data, energy_data, radius_data = load_sample_series(sample_name)

    fig = plt.figure(figsize=(6.0, 6.0), dpi=300)
    fig.patch.set_alpha(0.0)

    # Margins and layout matching reference style
    ax1 = fig.add_axes([0.15, 0.15, 0.70, 0.75])
    ax1.set_facecolor('none')
    ax1.set_box_aspect(1.0)

    # --- Left Y-Axis: Peak Energy (Blue) ---
    line1 = ax1.plot(
        x_data, energy_data,
        color='#1f77b4', marker='s', linewidth=1.0, markersize=3.0,
        label='Peak Energy'
    )
    ax1.set_ylabel('Peak Energy (eV)', color='#1f77b4', fontsize=14, fontname='Arial', labelpad=10)
    ax1.tick_params(axis='y', labelcolor='#1f77b4', direction='in', width=1.0, length=5, labelsize=12)
    ax1.tick_params(axis='x', direction='in', width=1.0, length=5, labelsize=12)

    for tick in ax1.get_yticklabels():
        tick.set_fontname('Arial')
    for tick in ax1.get_xticklabels():
        tick.set_fontname('Arial')

    # --- Right Y-Axis: Radius (Red) ---
    ax2 = ax1.twinx()
    ax2.set_facecolor('none')

    line2 = ax2.plot(
        x_data, radius_data,
        color='#d93838', marker='o', linewidth=1.0, markersize=3.0,
        label='Radius'
    )
    ax2.set_ylabel('Radius (nm)', color='#d93838', fontsize=14, fontname='Arial', labelpad=10)
    ax2.tick_params(axis='y', labelcolor='#d93838', direction='in', width=1.0, length=5, labelsize=12)

    for tick in ax2.get_yticklabels():
        tick.set_fontname('Arial')

    # Box / Frame styling on primary axis
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)

    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')
    for spine in ax2.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

    # X-axis label
    ax1.set_xlabel('Time (s)', fontsize=14, fontname='Arial', labelpad=10)

    # Save out high-res exact 1:1 square asset to Downloads folder using f-string
    safe_sample_name = sample_name.replace('-', '_')
    filename = rf"C:\Users\matth\Downloads\dual_axis_{safe_sample_name}_square.png"

    plt.savefig(
        filename,
        dpi=300,
        transparent=True,
        bbox_inches=None
    )
    print(f"✓ Saved plot for {sample_name} -> {filename}")
    plt.close(fig)

# =============================================================================
# MAIN RUNNER
# =============================================================================
if __name__ == "__main__":
    for sample in SAMPLES:
        try:
            plot_sample_dual_axis(sample)
        except Exception as e:
            print(f"[!] Error processing {sample}: {e}")
