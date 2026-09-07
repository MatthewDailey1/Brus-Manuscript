# -*- coding: utf-8 -*-
"""
Multi-Sample Kinetics Evaluation Suite (Square Format)
=============================================================================
1. Generates individual auto-fitted overlays for each material variant.
2. Combines all datasets into a single master summary comparison plot with fixed limits.
3. Automatically saves the master comparison plot into the user's Downloads folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure global matplotlib parameters for clean Arial formatting
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Configure GUI save behavior
plt.rcParams['savefig.transparent'] = True

# =============================================================================
# EXPLICIT FILE PATH & TRANSITION MARKER CONFIGURATION
# =============================================================================
MASTER_BRUS_EXCEL = r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\260717_Master_Data.xlsx"

SAMPLE_CONFIGS = [
    {
        "label": "Control",
        "lookup_key": "control",
        "scherrer_csv": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_Control_tube_S1_36_5min\processed_data\redo on this - better data\scherrer_kinetics_output-Control.csv",
        "brus_excel": MASTER_BRUS_EXCEL,
        "master_color": "#000000",  # Black
        "time_shift": 4.5
    },
    {
        "label": "3-APA",
        "lookup_key": "3apa",
        "scherrer_csv": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_APA_S1_30_tube_5min\processed_data\scherrer_kinetics_automated_output.csv",
        "brus_excel": MASTER_BRUS_EXCEL,
        "master_color": "#E07A5F",  # Coral / Terracotta
        "time_shift": 0.0
    },
    {
        "label": "4-ABA",
        "lookup_key": "4aba",
        "scherrer_csv": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_ABA_S1_30_tube_5min\processed_data_v2\scherrer_kinetics_automated_output.csv",
        "brus_excel": MASTER_BRUS_EXCEL,
        "master_color": "#38B000",  # Kelly Green
        "time_shift": 0.0
    },
    {
        "label": "5-AVA",
        "lookup_key": "5ava",
        "scherrer_csv": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_AVA_S1_18_5min -done\processed_data\scherrer_v2\scherrer_kinetics_output_AVA.csv",
        "brus_excel": MASTER_BRUS_EXCEL,
        "master_color": "#A381C6",  # Muted Purple
        "time_shift": 1.5
    },
    {
        "label": "7-AHA",
        "lookup_key": "7aha",
        "scherrer_csv": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_AHA_S1_30_tube_5min\processed_data_v2\scherrer_kinetics_automated_output.csv",
        "brus_excel": MASTER_BRUS_EXCEL,
        "master_color": "#70E4BC",  # Mint / Soft Teal
        "time_shift": 0.0
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
    # 1. Scherrer / GIWAXS Processing
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

    df_sch['time_clean'] = df_sch['time_clean'] + config["time_shift"]

    # 2. Brus/PL Processing
    df_brus = pd.read_excel(config["brus_excel"], sheet_name="RadiusResults_All")

    key = config["lookup_key"]
    t_brus_col = _find_col(df_brus, f'PL_FitResults_{key}__time_s', f'_{key}__time')
    rad_col = _find_col(df_brus, f'PL_FitResults_{key}__radius_nm', f'_{key}__radius')

    if t_brus_col is None: t_brus_col = _find_col(df_brus, 'time', key)
    if rad_col is None: rad_col = _find_col(df_brus, 'radius', 'nm', key)

    if t_brus_col is None or rad_col is None:
        raise ValueError(f"Could not map 3-header pattern for modifier '{key}' in 'RadiusResults_All'")

    df_brus['time_clean'] = pd.to_numeric(df_brus[t_brus_col], errors='coerce')
    df_brus['radius_clean'] = pd.to_numeric(df_brus[rad_col], errors='coerce')
    df_brus = df_brus.dropna(subset=['time_clean', 'radius_clean']).sort_values('time_clean')

    df_sch_vis = df_sch[(df_sch['time_clean'] >= 45) & (df_sch['time_clean'] <= 275)]
    df_brus_vis = df_brus[(df_brus['time_clean'] >= 45) & (df_brus['time_clean'] <= 275)]

    return df_sch_vis, df_brus_vis

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_individual_sample(config, df_sch_vis, df_brus_vis):
    """Generates an individual square plot (6.0\" x 6.0\")."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=300)

    # Set background transparency
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    color = config["master_color"]

    # 1. Scherrer (Lc): Pure scatter points, no line, no outline
    ax.scatter(df_sch_vis['time_clean'], df_sch_vis['scherrer_clean'],
               color=color, marker='o', s=15, linewidths=0, alpha=0.35, zorder=2,
               label='$D_{hkl}$')

    # 2. Brus (R_eff): Pure scatter points, no line, no outline
    ax.scatter(df_brus_vis['time_clean'], df_brus_vis['radius_clean'],
               color=color, marker='.', s=25, linewidths=0, alpha=1.0, zorder=3,
               label='$R_{\mathrm{eff}}$')

    # Axis Labels & Limits
    ax.set_xlabel('Time (s)', fontsize=18, fontname='Arial', labelpad=8)
    ax.set_ylabel('Characteristic Radius (nm)', fontsize=18, fontname='Arial', labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_xlim(45, 275)

    # Dynamic Ceiling & Ticks
    max_val = 10.0
    if len(df_sch_vis) > 0:
        max_val = max(max_val, df_sch_vis['scherrer_clean'].max())
    if len(df_brus_vis) > 0:
        max_val = max(max_val, df_brus_vis['radius_clean'].max())

    raw_ceiling = max_val * 1.15

    if raw_ceiling <= 12:
        y_ceiling = 12
    elif raw_ceiling <= 30:
        y_ceiling = int(3 * np.ceil(raw_ceiling / 3))
    else:
        y_ceiling = int(15 * np.ceil(raw_ceiling / 15))

    ax.set_ylim(0, y_ceiling)
    ax.set_yticks(np.linspace(0, y_ceiling, 4, dtype=int))

    # Styling ticks and spines
    ax.tick_params(axis='both', direction='in', top=True, right=True, width=1.5, length=6, labelsize=16)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.legend(loc='upper right', frameon=False, prop={'family': 'Arial', 'size': 14})

    # Optimized square layout margins
    plt.subplots_adjust(left=0.18, right=0.94, top=0.94, bottom=0.16)

    # Save PNG transparently to Downloads
    filename = r"C:\Users\matth\Downloads\sample_" + config['lookup_key'] + "_square_ppt.png"
    plt.savefig(filename, dpi=300, transparent=True, facecolor='none')

    # Display plot
    plt.show()


def plot_master_comparison(all_data):
    """Generates the compiled master square plot (6.0\" x 6.0\")."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=300)

    # Set background transparency
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    for label, info in all_data.items():
        df_sch = info["sch"]
        df_brus = info["brus"]
        color = info["color"]

        # Master Structural Track (Lc): Scatter points only
        ax.scatter(df_sch['time_clean'], df_sch['scherrer_clean'],
                   color=color, marker='o', s=12, linewidths=0, alpha=0.35, zorder=2,
                   label=f'{label} — ' + r'$D_{hkl}$')

        # Master Optical Track (R_eff): Scatter points only
        ax.scatter(df_brus['time_clean'], df_brus['radius_clean'],
                   color=color, marker='.', s=20, linewidths=0, alpha=1.0, zorder=3,
                   label=f'{label} — ' + r'$R_{\mathrm{eff}}$')

    # General Layout formatting & Y Label
    ax.set_xlabel('Time (s)', fontsize=18, fontname='Arial', labelpad=8)
    ax.set_ylabel('Characteristic Radius (nm)', fontsize=18, fontname='Arial', labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_xlim(45, 275)

    # Y-limits capped tightly at 60 with standard ticks
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 20, 40, 60])

    ax.tick_params(axis='both', direction='in', top=True, right=True, width=1.5, length=6, labelsize=16)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # COMPACT LEGEND SETTINGS
    ax.legend(
        loc='lower right',
        frameon=False,
        prop={'family': 'Arial', 'size': 11},
        labelspacing=0.3,
        handletextpad=0.4,
        ncol=1
    )

    # Optimized square layout margins
    plt.subplots_adjust(left=0.18, right=0.94, top=0.94, bottom=0.16)

    # Save Master Plot transparently directly to the Downloads folder
    master_filename = r"C:\Users\matth\Downloads\master_comparison_square_ppt.png"
    plt.savefig(master_filename, dpi=300, transparent=True, facecolor='none')
    print(f"✓ Successfully saved master comparison plot to: {master_filename}")

    # Display plot
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

            # Cache arrays for the composite compilation
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
