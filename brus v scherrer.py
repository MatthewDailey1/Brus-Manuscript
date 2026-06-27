# -*- coding: utf-8 -*-
"""
Multi-Sample Kinetics Evaluation Suite
=============================================================================
1. Generates individual publication-quality overlays for each material variant.
2. Combines all datasets into a single master summary comparison plot at the end.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure global matplotlib parameters for clean Arial formatting
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# =============================================================================
# EXPLICIT FILE PATH & TRANSITION MARKER CONFIGURATION
# =============================================================================
SAMPLE_CONFIGS = [
    {
        "label": "Control",
        "scherrer_csv": r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_Control_tube_S1_36_5min\processed_data\redo on this - better data\scherrer_kinetics_output-Control.csv",
        "brus_excel": r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_Control_tube_S1_36_5min\MAPI_Control_tube_S1_36_5min 003225 Spectrums\260623_Data\260623_Master_Data.xlsx",
        #"vlines": [50.0, 120.0, 210.0],
        "master_color": "blue"
    },
    {
        "label": "5-AVA",
        "scherrer_csv": r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_1pct_AVA_S1_18_5min -done\processed_data\scherrer_v2\scherrer_kinetics_output_AVA.csv",
        "brus_excel": r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_1pct_AVA_S1_18_5min -done\PL\260623_Data\260623_Master_Data.xlsx",
        #"vlines": [50.0, 120.0, 210.0],
        "master_color": "purple"
    },
    {
        "label": "5-AVAI",
        "scherrer_csv": r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_1pct_AVAI_S1_18_5min - done\processed_data\fitted data v2\scherrer_kinetics_automated_output-AVAI.csv",
        "brus_excel": r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_1pct_AVAI_S1_18_5min - done\PL\260623_Data\260623_Master_Data.xlsx",
        #"vlines": [50.0, 120.0, 210.0],
        "master_color": "forestgreen"
    },
    {
        "label": "5-AVACl",
        "scherrer_csv": r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_1pct_AVACl_S1_18_5min -PL good I think\processed_data\fitted data v2\scherrer_kinetics_automated_output-AVACl.csv",
        "brus_excel": r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_1pct_AVACl_S1_18_5min -PL good I think\PL\260623_Data\260623_Master_Data.xlsx",
        #"vlines": [50.0, 120.0, 210.0],
        "master_color": "crimson"
    }
]

# =============================================================================
# COLUMN SEARCH UTILITY
# =============================================================================

def _find_col(df, *keywords):
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in str(col).lower():
                return col
    return None

# =============================================================================
# DATA PIPELINE DATA EXTRACTION
# =============================================================================

def load_and_clean_sample(config):
    """Loads, shifts, and windows data frames for a single configuration."""
    # 1. Scherrer Processing
    df_sch = pd.read_csv(config["scherrer_csv"])
    t_sch_col = _find_col(df_sch, 'time_value', 'time', 'frame')
    size_col = _find_col(df_sch, 'scherrer_domain_size', 'size', 'scherrer', 'length')

    if t_sch_col is None or size_col is None:
        raise ValueError(f"Could not map Scherrer columns in: {config['scherrer_csv']}")

    df_sch['time_clean'] = pd.to_numeric(df_sch[t_sch_col], errors='coerce')
    df_sch['scherrer_clean'] = pd.to_numeric(df_sch[size_col], errors='coerce')
    df_sch = df_sch.dropna(subset=['time_clean', 'scherrer_clean']).sort_values('time_clean')

    if len(df_sch) > 15:
        time_offset = df_sch['time_clean'].iloc[15] - df_sch['time_clean'].iloc[0]
        df_sch['time_clean'] = df_sch['time_clean'] + time_offset

    # 2. Brus Processing
    df_brus = pd.read_excel(config["brus_excel"], sheet_name="RadiusResults_All")
    t_brus_col = _find_col(df_brus, 'time', 't_s', config["label"])
    rad_col = _find_col(df_brus, 'radius', 'rad', 'nm', config["label"])

    if t_brus_col is None or rad_col is None:
        raise ValueError(f"Could not map columns for label '{config['label']}' in 'RadiusResults_All'")

    df_brus['time_clean'] = pd.to_numeric(df_brus[t_brus_col], errors='coerce')
    df_brus['radius_clean'] = pd.to_numeric(df_brus[rad_col], errors='coerce')
    df_brus = df_brus.dropna(subset=['time_clean', 'radius_clean']).sort_values('time_clean')

    # Fixed typo here: changed df_brus_vis['time_clean'] to df_brus['time_clean'] on the right side
    df_sch_vis = df_sch[(df_sch['time_clean'] >= 45) & (df_sch['time_clean'] <= 275)]
    df_brus_vis = df_brus[(df_brus['time_clean'] >= 45) & (df_brus['time_clean'] <= 275)]

    return df_sch_vis, df_brus_vis

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_individual_sample(config, df_sch_vis, df_brus_vis):
    """Generates the clean individual designated color overlay plot starting at t=45s."""
    all_y = pd.concat([df_sch_vis['scherrer_clean'], df_brus_vis['radius_clean']])
    y_max = all_y.max() * 1.10 if not all_y.empty else 30

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Scherrer: Solid line with clean markers
    ax.plot(df_sch_vis['time_clean'], df_sch_vis['scherrer_clean'],
            color=config["master_color"], marker='o', markersize=3, linestyle='-', linewidth=1.5,
            label='Coherence Length ($L_c$)')

    # OPTION A: Clean Scatter Density Cloud instead of a line connection
    ax.scatter(df_brus_vis['time_clean'], df_brus_vis['radius_clean'],
               color=config["master_color"], marker='.', s=15, alpha=0.4,
               label='Brus Particle Radius ($R$)')

    # Phase boundaries
    if "vlines" in config and config["vlines"]:
        for xc in config["vlines"]:
            ax.axvline(x=xc, color="#555555", linestyle=':', linewidth=1.2, alpha=0.7,
                       label="_nolegend_", solid_capstyle='butt', dash_capstyle='butt')

    # Styling
    ax.set_xlabel('Time (s)', fontsize=13, fontname='Arial')
    ax.set_ylabel('Calculated Size (nm)', fontsize=13, fontname='Arial')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_xlim(45, 275)
    ax.set_ylim(0, y_max)

    ax.tick_params(axis='both', direction='in', top=True, right=True, width=1.5, length=6, labelsize=11)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.legend(loc='best', frameon=True, facecolor='white', shadow=False, edgecolor='#cccccc',
              prop={'family': 'Arial', 'size': 11})

    plt.tight_layout()
    plt.show()


def plot_master_comparison(all_data):
    """Generates the compiled multi-sample overlay graph starting at 45s,

    utilizing scatter tracking for the optical data to eliminate line noise.
    """
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)

    global_max = 30

    # Loop back through gathered sets and stack them onto one plot field
    for label, info in all_data.items():
        df_sch = info["sch"]
        df_brus = info["brus"]
        color = info["color"]

        all_y = pd.concat([df_sch['scherrer_clean'], df_brus['radius_clean']])
        if not all_y.empty:
            global_max = max(global_max, all_y.max())

        # Master Structural Track (Solid Line, small markers)
        ax.plot(df_sch['time_clean'], df_sch['scherrer_clean'],
                color=color, linestyle='-', linewidth=1.5, marker='o', markersize=3, alpha=0.85,
                label=f'{label} — Coherence Length ($L_c$)')

        # Master Optical Track: Configured as a scatter layer with pixel markers (marker='.')
        # and alpha blending to give a modern, clear distribution profile
        ax.scatter(df_brus['time_clean'], df_brus['radius_clean'],
                   color=color, marker='.', s=12, alpha=0.35, zorder=2,
                   label=f'{label} — Brus Radius ($R$)')

    # Apply shared global stage lines safely from configs if available
    shared_vlines = next((c["vlines"] for c in SAMPLE_CONFIGS if "vlines" in c), [])
    for xc in shared_vlines:
        ax.axvline(x=xc, color="#555555", linestyle=':', linewidth=1.2, alpha=0.5,
                   label="_nolegend_", solid_capstyle='butt', dash_capstyle='butt')

    # General Layout formatting
    ax.set_xlabel('Time (s)', fontsize=13, fontname='Arial')
    ax.set_ylabel('Calculated Size (nm)', fontsize=13, fontname='Arial')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_xlim(45, 275)

    # Headroom Buffer to clear the legend box safely over your data peaks
    ax.set_ylim(0, global_max * 1.32)

    ax.tick_params(axis='both', direction='in', top=True, right=True, width=1.5, length=6, labelsize=11)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.legend(loc='upper right', frameon=True, facecolor='white', shadow=False, edgecolor='#cccccc',
              prop={'family': 'Arial', 'size': 9.5}, ncol=1)

    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN RUNNER EXECUTION
# =============================================================================

def main():
    compiled_datasets = {}

    # Stage 1: Individual Iterative Runs
    for config in SAMPLE_CONFIGS:
        if not os.path.exists(config["scherrer_csv"]) or not os.path.exists(config["brus_excel"]):
            print(f"[!] Paths unreachable for sample: {config['label']}. Skipping.")
            continue

        try:
            df_sch_vis, df_brus_vis = load_and_clean_sample(config)

            # Print separate plot
            plot_individual_sample(config, df_sch_vis, df_brus_vis)
            print(f"✓ Completed isolated display window for: {config['label']}")

            # Cache the arrays for the big finale
            compiled_datasets[config["label"]] = {
                "sch": df_sch_vis,
                "brus": df_brus_vis,
                "color": config["master_color"]
            }
        except Exception as e:
            print(f"[!] Processing broke down on {config['label']}: {e}")

    # Stage 2: Combined Master Plot Execution
    if compiled_datasets:
        print("\n--> Generating Master Combined Comparison Overlay plot...")
        plot_master_comparison(compiled_datasets)
        print("✓ Done.")

if __name__ == "__main__":
    main()
