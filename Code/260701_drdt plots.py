import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from pathlib import Path

# Clean classic/scientific look
plt.style.use('seaborn-v0_8-whitegrid')

# =========================================================================
# EXPERIMENTAL CONFIGURATION
# =========================================================================
SAMPLE_ORDER = ['control', '3-APA', '4-ABA', '5-AVA', '7-AHA']

SAMPLE_STYLES = {
    'control': {'color': '#000000', 'label': 'Control'},  # Black
    '3-APA'  : {'color': '#E07A5F', 'label': '3-APA'},    # Coral / Terracotta
    '4-ABA'  : {'color': '#38B000', 'label': '4-ABA'},    # Kelly Green
    '5-AVA'  : {'color': '#A381C6', 'label': '5-AVA'},    # Muted Purple
    '7-AHA'  : {'color': '#70E4BC', 'label': '7-AHA'}     # Mint / Soft Teal
}

SAMPLE_HEADER_KEYS = {
    'control': 'control',
    '3-APA':   '3APA',
    '4-ABA':   '4ABA',
    '5-AVA':   '5AVA',
    '7-AHA':   '7AHA'
}

MASTER_FILE_PATH = r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\260717_Master_Data.xlsx"
TARGET_SHEET = 'GrowthRate_All'
OUTPUT_DIR = r"C:\Users\matth\Downloads"

master_summary_data = {}

# =========================================================================
# LOAD DATASET
# =========================================================================
path_obj = Path(MASTER_FILE_PATH)

if path_obj.exists():
    try:
        df_master = pd.read_excel(path_obj, sheet_name=TARGET_SHEET, engine='openpyxl')
        for sample_key in SAMPLE_ORDER:
            header_id = SAMPLE_HEADER_KEYS.get(sample_key)
            sample_cols = [col for col in df_master.columns if header_id in str(col)]
            if len(sample_cols) >= 6:
                master_summary_data[sample_key] = df_master[sample_cols].copy()
    except Exception as e:
        print(f"[!] Error loading excel: {e}")

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

# =========================================================================
# HELPER: APPLY AXIS & SPINE STYLING
# =========================================================================
def style_axis(ax, y_max=1.2):
    ax.set_xlabel("Time (s)", fontsize=20, labelpad=8)
    ax.set_ylabel("d(Radius)/dt (nm/s)", fontsize=20, labelpad=8)

    ax.set_xlim(45, 150)
    ax.set_ylim(-0.02, y_max)

    ax.set_xticks([50, 75, 100, 125, 150])

    # Generate 4 evenly spaced tick marks up to y_max
    y_ticks = np.linspace(0.0, np.round(y_max, 2), 4)
    ax.set_yticks(np.round(y_ticks, 2))

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.8)

    ax.tick_params(
        direction='in',
        top=True, right=True, bottom=True, left=True,
        length=7,
        width=1.8,
        colors='black',
        labelsize=18
    )

# =========================================================================
# PLOT A SINGLE SAMPLE
# =========================================================================
def plot_sample_lines(ax, df, style):
    legend_labeled = False
    max_y_value = 0.0

    for i in range(0, 6, 2):
        x_col_idx = i
        y_col_idx = i + 1

        plot_data = df.iloc[:, [x_col_idx, y_col_idx]].dropna()

        if not plot_data.empty:
            region_sorted = plot_data.sort_values(by=plot_data.columns[0])
            x_vals = region_sorted.iloc[:, 0].values
            y_vals = region_sorted.iloc[:, 1].values

            if i == 0 and len(x_vals) > 0:
                valid_mask = x_vals <= 80
                x_vals = x_vals[valid_mask]
                y_vals = y_vals[valid_mask]

            elif i == 2 and len(x_vals) > 0:
                valid_mask = (x_vals > 80) & (x_vals <= 105)
                x_vals = x_vals[valid_mask]
                y_vals = y_vals[valid_mask]

            elif i == 4 and len(x_vals) > 0:
                valid_mask = x_vals >= 105
                x_vals = x_vals[valid_mask]
                y_vals = y_vals[valid_mask]

            if len(y_vals) > 0:
                max_y_value = max(max_y_value, y_vals.max())

            current_label = style['label'] if not legend_labeled else ""
            if current_label:
                legend_labeled = True

            ax.plot(
                x_vals, y_vals,
                linestyle='-.',
                linewidth=2.5,
                color=style['color'],
                alpha=0.95,
                label=current_label
            )

    return max_y_value

# =========================================================================
# MAIN EXECUTION
# =========================================================================
def main():
    # 1. GENERATE INDIVIDUAL PLOTS (With auto-tightened Y-limits)
    for sample_key in SAMPLE_ORDER:
        if sample_key in master_summary_data:
            fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
            fig.patch.set_alpha(0.0)
            ax.set_facecolor('none')

            df = master_summary_data[sample_key]
            style = SAMPLE_STYLES[sample_key]

            max_y_val = plot_sample_lines(ax, df, style)

            # Add a 10% ceiling buffer above maximum observed data value
            tight_y_ceiling = max_y_val * 1.10 if max_y_val > 0 else 1.2

            style_axis(ax, y_max=tight_y_ceiling)

            ax.legend(loc='upper right', frameon=False, fontsize=15)

            out_path = os.path.join(OUTPUT_DIR, f"growth_rate_{sample_key}_square.png")
            save_square_plot(fig, out_path)
            plt.show()

    # 2. GENERATE COMBINED SUMMARY PLOT (Standard 1.2 Y-limit)
    if master_summary_data:
        print("\n--> Generating Combined Master Growth Rate Plot...")
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')

        for sample_key in SAMPLE_ORDER:
            if sample_key in master_summary_data:
                df = master_summary_data[sample_key]
                style = SAMPLE_STYLES[sample_key]
                plot_sample_lines(ax, df, style)

        style_axis(ax, y_max=1.2)
        ax.legend(loc='upper right', frameon=False, fontsize=15)

        master_out_path = os.path.join(OUTPUT_DIR, "growth_rate_master_combined_square.png")
        save_square_plot(fig, master_out_path)
        plt.show()

if __name__ == "__main__":
    main()
