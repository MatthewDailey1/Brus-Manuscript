# -*- coding: utf-8 -*-
"""
Multi-Sample Kinetics Evaluation Suite (Square Format)
=============================================================================
1. Generates individual auto-fitted overlays for each material variant.
2. Combines all datasets into a single master summary comparison plot.
3. For the Brus selection, automatically exports 4 distinct style variations:
   - Dashed-dot line without t0/tf dots
   - Dashed-dot line with t0/tf dots
   - Solid line without t0/tf dots
   - Solid line with t0/tf dots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

# Configure global matplotlib parameters for clean Arial formatting
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Configure GUI save behavior
plt.rcParams['savefig.transparent'] = True

# =============================================================================
# EXPLICIT FILE PATH & TRANSITION MARKER CONFIGURATION
# =============================================================================
MASTER_BRUS_EXCEL = r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\260717_Master_Data.xlsx"

# Sample-specific time bounds for t0 and tf (seconds)
SAMPLE_TIME_BOUNDS = {
    "Control": {"t0": 83.0, "tf": 99.0},
    "3-APA":   {"t0": 83.0, "tf": 101.0},
    "4-ABA":   {"t0": 84.0, "tf": 100.0},
    "5-AVA":   {"t0": 84.0, "tf": 100.0},
    "7-AHA":   {"t0": 83.0, "tf": 100.0}
}

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
# COLUMN SEARCH & INTERPOLATION UTILITIES
# =============================================================================

def _find_col(df, *keywords):
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in str(col).lower():
                return col
    return None

def extract_value_at_exact_time(x_arr, y_arr, target_time):
    """Interpolates y-value at exact target_time."""
    if len(x_arr) == 0:
        return None
    return np.interp(target_time, x_arr, y_arr)

# =============================================================================
# DATA PIPELINE EXTRACTION
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

# =========================================================================
# HELPER: CROP & SAVE AS EXPLICIT SQUARE
# =========================================================================

def save_square_plot(fig, filename):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_bbox = fig.get_tightbbox(renderer)

    pad = 0.04
    tight_bbox = tight_bbox.padded(pad)

    content_w, content_h = tight_bbox.width, tight_bbox.height
    side = max(content_w, content_h)

    x0 = tight_bbox.x0 - (side - content_w) / 2.0
    y0 = tight_bbox.y0 - (side - content_h) / 2.0
    square_bbox = Bbox.from_bounds(x0, y0, side, side)

    plt.savefig(
        filename,
        dpi=300,
        transparent=True,
        facecolor='none',
        bbox_inches=square_bbox
    )
    print(f"[✓] Saved square plot: {filename}")

# =============================================================================
# CORE RENDERING ENGINES FOR EXPORT VARIATIONS
# =============================================================================

def render_single_individual_plot(config, df_sch_vis, df_brus_vis, plot_choice, linestyle='-.', show_dots=False, suffix=""):
    """Renders a single individual plot instance with specific line style and dot settings."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=300)

    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    color = config["master_color"]
    label = config["label"]

    # 1. Scherrer Track
    if plot_choice in ['scherrer', 'both']:
        ax.plot(df_sch_vis['time_clean'], df_sch_vis['scherrer_clean'],
                color=color, linestyle='-.', linewidth=1.0, alpha=0.35, zorder=2,
                label='$D_{hkl}$')

    # 2. Brus Track
    if plot_choice in ['brus', 'both']:
        line_style_to_use = linestyle if plot_choice == 'brus' else '-.'
        ax.plot(df_brus_vis['time_clean'], df_brus_vis['radius_clean'],
                color=color, linestyle=line_style_to_use, linewidth=1.0, alpha=1.0, zorder=3,
                label='$R_{\mathrm{eff}}$')

        # Scatter dots at t0 and tf if requested
        if plot_choice == 'brus' and show_dots and (label in SAMPLE_TIME_BOUNDS):
            t0 = SAMPLE_TIME_BOUNDS[label]["t0"]
            tf = SAMPLE_TIME_BOUNDS[label]["tf"]

            r_t0 = extract_value_at_exact_time(df_brus_vis['time_clean'].values, df_brus_vis['radius_clean'].values, t0)
            r_tf = extract_value_at_exact_time(df_brus_vis['time_clean'].values, df_brus_vis['radius_clean'].values, tf)

            if r_t0 is not None:
                ax.scatter(t0, r_t0, color=color, s=30, zorder=5, edgecolor='none')
            if r_tf is not None:
                ax.scatter(tf, r_tf, color=color, s=30, zorder=5, edgecolor='none')

    # Formatting & Limits
    ax.set_xlabel('Time (s)', fontsize=18, fontname='Arial', labelpad=8)
    ax.set_ylabel('Characteristic Radius (nm)', fontsize=18, fontname='Arial', labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.3)

    ax.set_xlim(75, 115)
    ax.set_xticks([80, 90, 100, 110])

    if plot_choice == 'brus':
        ax.set_ylim(4, 14)
        ax.set_yticks([6, 8, 10, 12, 14])
    else:
        max_val = 10.0
        if plot_choice in ['scherrer', 'both'] and len(df_sch_vis) > 0:
            max_val = max(max_val, df_sch_vis['scherrer_clean'].max())
        if plot_choice in ['brus', 'both'] and len(df_brus_vis) > 0:
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

    ax.tick_params(axis='both', direction='in', top=True, right=True, width=1.0, length=6, labelsize=16, zorder=10)
    for label_obj in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_obj.set_fontname('Arial')
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_zorder(10)

    ax.legend(loc='upper right', frameon=False, prop={'family': 'Arial', 'size': 18})

    filename = r"C:\Users\matth\Downloads\sample_" + config['lookup_key'] + f"_square_ppt_{plot_choice}{suffix}.png"
    save_square_plot(fig, filename)
    plt.close(fig)


def render_single_master_plot(all_data, plot_choice, linestyle='-.', show_dots=False, suffix=""):
    """Renders a single master comparison plot instance with specific line style and dot settings."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=300)

    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    for label, info in all_data.items():
        df_sch = info["sch"]
        df_brus = info["brus"]
        color = info["color"]

        # Master Structural Track
        if plot_choice in ['scherrer', 'both']:
            sch_label = f'{label} — ' + r'$D_{hkl}$' if plot_choice == 'both' else label
            ax.plot(df_sch['time_clean'], df_sch['scherrer_clean'],
                    color=color, linestyle='-.', linewidth=1.0, alpha=0.35, zorder=2,
                    label=sch_label)

        # Master Optical Track
        if plot_choice in ['brus', 'both']:
            brus_label = f'{label} — ' + r'$R_{\mathrm{eff}}$' if plot_choice == 'both' else label
            line_style_to_use = linestyle if plot_choice == 'brus' else '-.'

            ax.plot(df_brus['time_clean'], df_brus['radius_clean'],
                    color=color, linestyle=line_style_to_use, linewidth=1.0, alpha=1.0, zorder=3,
                    label=brus_label)

            # Scatter dots at t0 and tf if requested
            if plot_choice == 'brus' and show_dots and (label in SAMPLE_TIME_BOUNDS):
                t0 = SAMPLE_TIME_BOUNDS[label]["t0"]
                tf = SAMPLE_TIME_BOUNDS[label]["tf"]

                r_t0 = extract_value_at_exact_time(df_brus['time_clean'].values, df_brus['radius_clean'].values, t0)
                r_tf = extract_value_at_exact_time(df_brus['time_clean'].values, df_brus['radius_clean'].values, tf)

                if r_t0 is not None:
                    ax.scatter(t0, r_t0, color=color, s=30, zorder=5, edgecolor='none')
                if r_tf is not None:
                    ax.scatter(tf, r_tf, color=color, s=30, zorder=5, edgecolor='none')

    # Formatting & Limits
    ax.set_xlabel('Time (s)', fontsize=18, fontname='Arial', labelpad=8)
    ax.set_ylabel('$R_{\mathrm{eff}}$ (nm)', fontsize=18, fontname='Arial', labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.3)

    ax.set_xlim(75, 115)
    ax.set_xticks([80, 90, 100, 110])

    if plot_choice == 'brus':
        ax.set_ylim(4, 14)
        ax.set_yticks([6, 8, 10, 12, 14])
    else:
        ax.set_ylim(0, 60)
        ax.set_yticks([0, 20, 40, 60])

    ax.tick_params(axis='both', direction='in', top=True, right=True, width=1.5, length=6, labelsize=16, zorder=10)
    for label_obj in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_obj.set_fontname('Arial')
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_zorder(10)

    ax.legend(
        loc='lower right',
        frameon=False,
        prop={'family': 'Arial', 'size': 11},
        labelspacing=0.3,
        handletextpad=0.4,
        ncol=1
    )

    master_filename = r"C:\Users\matth\Downloads\master_comparison_square_ppt_" + plot_choice + f"{suffix}.png"
    save_square_plot(fig, master_filename)
    plt.close(fig)

# =============================================================================
# EXPORT MANAGERS
# =============================================================================

def plot_individual_sample(config, df_sch_vis, df_brus_vis, plot_choice):
    """Executes the 4 export variations for individual samples if Brus is chosen."""
    if plot_choice == 'brus':
        # 1. Dashed-dot, No dots
        render_single_individual_plot(config, df_sch_vis, df_brus_vis, plot_choice, linestyle='-.', show_dots=False, suffix="_dashed_nodots")
        # 2. Dashed-dot, With dots
        render_single_individual_plot(config, df_sch_vis, df_brus_vis, plot_choice, linestyle='-.', show_dots=True, suffix="_dashed_dots")
        # 3. Solid, No dots
        render_single_individual_plot(config, df_sch_vis, df_brus_vis, plot_choice, linestyle='-', show_dots=False, suffix="_solid_nodots")
        # 4. Solid, With dots
        render_single_individual_plot(config, df_sch_vis, df_brus_vis, plot_choice, linestyle='-', show_dots=True, suffix="_solid_dots")
    else:
        render_single_individual_plot(config, df_sch_vis, df_brus_vis, plot_choice, linestyle='-.', show_dots=False, suffix="")


def plot_master_comparison(all_data, plot_choice):
    """Executes the 4 export variations for the master plot if Brus is chosen."""
    if plot_choice == 'brus':
        print("  --> Exporting 1/4: Dashed line, No dots...")
        render_single_master_plot(all_data, plot_choice, linestyle='-.', show_dots=False, suffix="_dashed_nodots")
        print("  --> Exporting 2/4: Dashed line, With dots...")
        render_single_master_plot(all_data, plot_choice, linestyle='-.', show_dots=True, suffix="_dashed_dots")
        print("  --> Exporting 3/4: Solid line, No dots...")
        render_single_master_plot(all_data, plot_choice, linestyle='-', show_dots=False, suffix="_solid_nodots")
        print("  --> Exporting 4/4: Solid line, With dots...")
        render_single_master_plot(all_data, plot_choice, linestyle='-', show_dots=True, suffix="_solid_dots")
    else:
        render_single_master_plot(all_data, plot_choice, linestyle='-.', show_dots=False, suffix="")

# =============================================================================
# MAIN RUNNER EXECUTION
# =============================================================================

def main():
    print("Which data would you like to plot?")
    print("1. Scherrer (Only Structural)")
    print("2. Brus (Only Optical)")
    print("3. Both")

    while True:
        choice = input("Enter your choice (Scherrer/Brus/Both) [Default: Both]: ").strip().lower()
        if choice in ['scherrer', 'brus', 'both']:
            break
        elif choice == '':
            choice = 'both'
            break
        else:
            print("Invalid input. Please type 'Scherrer', 'Brus', or 'Both'.")

    compiled_datasets = {}

    # Stage 1: Individual Iterative Runs
    for config in SAMPLE_CONFIGS:
        if not os.path.exists(config["scherrer_csv"]) or not os.path.exists(config["brus_excel"]):
            print(f"[!] Paths unreachable for sample: {config['label']}. Skipping.")
            continue

        try:
            df_sch_vis, df_brus_vis = load_and_clean_sample(config)

            plot_individual_sample(config, df_sch_vis, df_brus_vis, choice)

            compiled_datasets[config["label"]] = {
                "sch": df_sch_vis,
                "brus": df_brus_vis,
                "color": config["master_color"]
            }
        except Exception as e:
            print(f"[!] Processing broke down on {config['label']}: {e}")

    # Stage 2: Combined Master Plot Execution
    if compiled_datasets:
        print("\n--> Generating Master Combined Comparison Overlay exports...")
        plot_master_comparison(compiled_datasets, choice)

if __name__ == "__main__":
    main()
