# -*- coding: utf-8 -*-
"""
Integrated Master Nanoparticle Kinetic & Descriptor Suite
=============================================================================
1. Figure a: PL Emission Onset line-dot plot:
   - Equal categorical sample spacing on X-axis.
   - Y-axis from 38 to 58 with clean integer steps [38, 42, 46, 50, 54, 58].
   - Clean inward tick marks.
2. Rate Charts: Square initial rate (t0) and paired (t0 vs tf) bar charts
   with clean inward ticks.
3. Figure c: Rate-Extent Tradeoff using Brus effective radius curves:
   - Y-axis: Delta Reff = Reff(tf) - Reff(t0)
   - X-axis: Elapsed t_1/2 to reach 50% Delta Reff relative to t0
   - Organic axis auto-scaling so Control and all additives remain visible.

All exported PNG files are strictly forced into 1:1 square bounding boxes.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
from docx import Document

# Global typography setup
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# =============================================================================
# GLOBAL EXPERIMENTAL CONFIGURATION
# =============================================================================
OUTPUT_DIR = r"C:\Users\matth\Downloads"
MASTER_EXCEL_PATH = r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\260717_Master_Data.xlsx"
RATE_SHEET = 'GrowthRate_All'
RADIUS_SHEET = 'RadiusResults_All'

PL_SAMPLES = [
    {
        "label": "Control",
        "header_key": "control",
        "lookup_key": "control",
        "docx_path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\Reports\260717_PL_FitResults_control.docx",
        "color": "#232323",  # Deep Charcoal / Black
        "carbon_num": 0
    },
    {
        "label": "3-APA",
        "header_key": "3APA",
        "lookup_key": "3apa",
        "docx_path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\Reports\260717_PL_FitResults_3APA.docx",
        "color": "#E07A5F",  # Coral / Terracotta
        "carbon_num": 3
    },
    {
        "label": "4-ABA",
        "header_key": "4ABA",
        "lookup_key": "4aba",
        "docx_path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\Reports\260717_PL_FitResults_4ABA.docx",
        "color": "#38B000",  # Kelly Green
        "carbon_num": 4
    },
    {
        "label": "5-AVA",
        "header_key": "5AVA",
        "lookup_key": "5ava",
        "docx_path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\Reports\260717_PL_FitResults_5AVA.docx",
        "color": "#A381C6",  # Muted Purple
        "carbon_num": 5
    },
    {
        "label": "7-AHA",
        "header_key": "7AHA",
        "lookup_key": "7aha",
        "docx_path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\Reports\260717_PL_FitResults_7AHA.docx",
        "color": "#70E4BC",  # Mint / Soft Teal
        "carbon_num": 7
    }
]

# Sample-specific time bounds for growth rates & Brus radius (seconds)
SAMPLE_TIME_BOUNDS = {
    "Control": {"t0": 83.0, "tf": 99.0},
    "3-APA":   {"t0": 83.0, "tf": 101.0},
    "4-ABA":   {"t0": 84.0, "tf": 100.0},
    "5-AVA":   {"t0": 84.0, "tf": 100.0},
    "7-AHA":   {"t0": 83.0, "tf": 100.0}
}

# =============================================================================
# EXPLICIT 1:1 SQUARE SAVING ENGINE
# =============================================================================
def save_square_plot(fig, filepath):
    """Forces the exported figure file to have a strict 1:1 square aspect ratio."""
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
        filepath,
        dpi=300,
        transparent=True,
        facecolor='none',
        bbox_inches=square_bbox
    )
    print(f"  [✓] Exported strict square figure: {filepath}")

# =============================================================================
# DATA EXTRACTION HELPERS (.DOCX & EXCEL)
# =============================================================================
def _find_col(df, *keywords):
    """Utility to map flexible column names in Excel sheets."""
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in str(col).lower():
                return col
    return None

def parse_docx_region_bounds(docx_path):
    """Parses a Word document (.docx) to extract Region 1 start timestamps."""
    doc = Document(docx_path)
    all_lines = []

    for p in doc.paragraphs:
        if p.text.strip():
            all_lines.append(p.text.strip())

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    all_lines.append(cell.text.strip())

    full_text = "\n".join(all_lines)

    def find_region_range(region_num):
        pattern = rf"Region\s*{region_num}[:\s|-]*([\d\.]+)\s*to\s*([\d\.]+)"
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            return float(match.group(1)), float(match.group(2))
        return 0.0, 0.0

    r1_start, _ = find_region_range(1)
    return {'r1_start': r1_start}

def batch_process_docx_files():
    """Loops through sample configs and reads Region 1 start bounds from .docx files."""
    results = []
    for cfg in PL_SAMPLES:
        try:
            bounds = parse_docx_region_bounds(cfg['docx_path'])
            bounds['label'] = cfg['label']
            bounds['color'] = cfg['color']
            bounds['carbon_num'] = cfg['carbon_num']
            results.append(bounds)
            print(f"  ✓ Parsed {cfg['label']}: Onset @ {bounds['r1_start']}s")
        except Exception as e:
            print(f"  [!] Could not parse .docx for {cfg['label']} ({cfg['docx_path']}): {e}")

    return pd.DataFrame(results)

def get_sample_xy_rate_data(df_rates, header_key):
    """Extracts rate data from the Master Excel GrowthRate_All sheet."""
    sample_cols = [col for col in df_rates.columns if header_key in str(col)]
    if len(sample_cols) < 2:
        return np.array([]), np.array([])

    df_sample = df_rates[sample_cols]

    x_list, y_list = [], []
    for i in range(0, len(sample_cols), 2):
        if i + 1 < len(sample_cols):
            plot_data = df_sample.iloc[:, [i, i+1]].dropna()
            if not plot_data.empty:
                x_list.append(plot_data.iloc[:, 0])
                y_list.append(plot_data.iloc[:, 1])

    if not x_list:
        return np.array([]), np.array([])

    all_x = pd.concat(x_list).values
    all_y = pd.concat(y_list).values

    sort_idx = np.argsort(all_x)
    return all_x[sort_idx], all_y[sort_idx]

def get_sample_brus_radius_data(df_brus, lookup_key):
    """Extracts Brus effective radius data from RadiusResults_All sheet."""
    key = lookup_key
    t_col = _find_col(df_brus, f'PL_FitResults_{key}__time_s', f'_{key}__time', 'time', key)
    r_col = _find_col(df_brus, f'PL_FitResults_{key}__radius_nm', f'_{key}__radius', 'radius', 'nm', key)

    if t_col is None or r_col is None:
        return np.array([]), np.array([])

    df_clean = df_brus.copy()
    df_clean['t'] = pd.to_numeric(df_clean[t_col], errors='coerce')
    df_clean['r'] = pd.to_numeric(df_clean[r_col], errors='coerce')
    df_clean = df_clean.dropna(subset=['t', 'r']).sort_values('t')

    return df_clean['t'].values, df_clean['r'].values

def extract_value_at_exact_time(x_arr, y_arr, target_time):
    """Interpolates value at a specific timestamp."""
    if len(x_arr) == 0: return 0.0
    return np.interp(target_time, x_arr, y_arr)

# =============================================================================
# PLOTTING ROUTINES (STRICT SQUARE CANVAS & CLEAN INWARD TICKS)
# =============================================================================
def plot_pl_onset_connected_line(df_results, save_filename="fig_a_pl_onset_line_dots.png"):
    """
    Generates Figure a in strict square format:
    - Equal categorical sample spacing on X-axis with padded bounds.
    - Y-axis from 38 to 58 with clean integer steps.
    - Clean inward tick marks.
    """
    print("--> Generating Square PL Emission Onset Graph (Figure a)...")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

    df_sorted = df_results.sort_values(by='carbon_num').reset_index(drop=True)

    x_coords = np.arange(len(df_sorted))
    y_vals = df_sorted['r1_start'].values

    # Connecting trendline & scatter dots
    ax.plot(x_coords, y_vals, color="#000000", linewidth=1.0, zorder=1)
    for idx, row in df_sorted.iterrows():
        ax.scatter(x_coords[idx], row['r1_start'], color=row['color'],
                   s=90, zorder=2, edgecolor='none')

    # --- FIX FOR EDGE COLLISION ---
    # Padding the x-limits gives the outer dots half a step of breathing room
    ax.set_xlim(-0.5, len(df_sorted) - 0.5)

    ax.set_xticks(x_coords)
    ax.set_xticklabels(df_sorted['label'], fontsize=14)

    # Y-AXIS: 38 to 58 with clean integer steps
    ax.set_ylim(38, 58)
    ax.set_yticks([38, 42, 46, 50, 54, 58])

    # CLEAN INWARD TICKS
    ax.tick_params(axis='both', direction='in', bottom=True, left=True, top=False, right=False, labelsize=12, width=1.0, length=5)

    ax.set_xlabel("Amino-acid carbon number", fontsize=18, labelpad=8)
    ax.set_ylabel("PL Onset Time (s)", fontsize=18)
    ax.set_box_aspect(1)

    save_square_plot(fig, os.path.join(OUTPUT_DIR, save_filename))
    plt.show()
    plt.close(fig)

def generate_growth_rate_charts(df_rates):
    """Generates initial rate (t0) and side-by-side grouped (t0 vs tf) rate charts with clean inward ticks."""
    sample_data = {}
    for cfg in PL_SAMPLES:
        x, y = get_sample_xy_rate_data(df_rates, cfg['header_key'])
        sample_data[cfg['label']] = (x, y)

    labels = [cfg['label'] for cfg in PL_SAMPLES]
    colors = [cfg['color'] for cfg in PL_SAMPLES]
    x_coords = np.arange(len(labels))

    bar_width = 0.32
    bar_offset = 0.175

    early_rates = []
    late_rates = []

    for cfg in PL_SAMPLES:
        lbl = cfg['label']
        x, y = sample_data[lbl]
        t0 = SAMPLE_TIME_BOUNDS[lbl]["t0"]
        tf = SAMPLE_TIME_BOUNDS[lbl]["tf"]

        early_rates.append(extract_value_at_exact_time(x, y, target_time=t0))
        late_rates.append(extract_value_at_exact_time(x, y, target_time=tf))

    # --- INITIAL RATE CHART (t0) ---
    print("--> Generating Square Initial Growth Rate Graph (t0)...")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.bar(x_coords, early_rates, width=0.6, color=colors, edgecolor='none')
    ax.set_xticks(x_coords)
    ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(axis='both', direction='in', bottom=True, left=True, top=False, right=False, labelsize=12, width=1.0, length=5)
    ax.set_ylabel("Initial Growth Rate d(Radius)/dt (nm s$^{-1}$)", fontsize=14)
    ax.set_box_aspect(1)

    save_square_plot(fig, os.path.join(OUTPUT_DIR, "pub_square_t0_rates.png"))
    plt.show()
    plt.close(fig)

    # --- GROUPED SIDE-BY-SIDE CHART (t0 vs tf) ---
    print("--> Generating Square Side-by-Side Growth Rate Graph (t0 vs tf)...")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

    ax.bar(x_coords - bar_offset, early_rates, bar_width, color=colors, edgecolor='none')
    ax.bar(x_coords + bar_offset, late_rates, bar_width, color=colors, alpha=0.6, hatch='//', edgecolor='white', linewidth=0)

    ax.set_xticks(x_coords)
    ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(axis='both', direction='in', bottom=True, left=True, top=False, right=False, labelsize=14, width=1.0, length=5)
    ax.set_ylabel("d(Radius)/dt (nm s$^{-1}$)", fontsize=18)
    ax.set_box_aspect(1)

    t0_patch = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=1.0, label='$t_0$ @ 25 °C')
    tf_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', linewidth=1.0, label='$t_f$ @ 100 °C')
    ax.legend(handles=[t0_patch, tf_patch], frameon=False, fontsize=14)

    save_square_plot(fig, os.path.join(OUTPUT_DIR, "pub_square_side_by_side_rates.png"))
    plt.show()
    plt.close(fig)

def plot_rate_extent_tradeoff(df_radius, save_filename="fig_c_rate_extent_tradeoff.png"):
    """
    Generates Figure c (Rate-Extent Tradeoff):
    - Uses organic auto-scaling bounds so Control (~16-18s) and all additives stay inside.
    - Clean inward ticks.
    - Offsets the Control label to the left so it never clips past the right border.
    """
    print("--> Generating Square Rate-Extent Tradeoff Graph from Brus Reff (Figure c)...")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

    x_points, y_points = [], []

    for cfg in PL_SAMPLES:
        lbl = cfg['label']
        col = cfg['color']

        t0 = SAMPLE_TIME_BOUNDS[lbl]["t0"]
        tf = SAMPLE_TIME_BOUNDS[lbl]["tf"]

        # Extract Brus radius (Reff) vs time curve
        x_arr, y_arr = get_sample_brus_radius_data(df_radius, cfg['lookup_key'])
        if len(x_arr) == 0:
            print(f"  [!] Missing Brus radius data for {lbl}. Skipping.")
            continue

        r_t0 = extract_value_at_exact_time(x_arr, y_arr, t0)
        r_tf = extract_value_at_exact_time(x_arr, y_arr, tf)

        # 1. Total Brus Radius Growth Extent (Y-axis)
        delta_reff = r_tf - r_t0

        # 2. Target radius value at 50% growth progress
        r_half_target = r_t0 + 0.5 * delta_reff

        # 3. Crop trace to sample-specific window (t0 -> tf)
        mask = (x_arr >= t0) & (x_arr <= tf)
        x_win, y_win = x_arr[mask], y_arr[mask]

        # Interpolate exact absolute timestamp where radius hits 50% height
        if len(x_win) > 1:
            t_half_abs = np.interp(r_half_target, y_win, x_win)
            t_half = t_half_abs - t0  # Elapsed seconds from t0 to reach 50% growth
        else:
            t_half = 0.0

        x_points.append(t_half)
        y_points.append(delta_reff)

        # Plot scatter point
        ax.scatter(t_half, delta_reff, color=col, s=100, zorder=3, edgecolor='none')

        # --- LABEL ANNOTATIONS ---
        # Shift Control label to the left of its marker so it stays within frame
        if lbl == "Control":
            ax.annotate(lbl, (t_half, delta_reff), xytext=(-8, 2),
                        textcoords='offset points', ha='right', fontsize=11, fontname='Arial')
        else:
            ax.annotate(lbl, (t_half, delta_reff), xytext=(6, 2),
                        textcoords='offset points', ha='left', fontsize=11, fontname='Arial')

    # ORGANIC AXIS AUTO-SCALING (Ensures Control is never cropped)
    if x_points and y_points:
        x_min, x_max = min(x_points), max(x_points)
        y_min, y_max = min(y_points), max(y_points)

        x_pad = (x_max - x_min) * 0.12 if x_max != x_min else 1.0
        y_pad = (y_max - y_min) * 0.12 if y_max != y_min else 0.5

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel("Effective half-time, $t_{1/2}$ (s)", fontsize=18)
    ax.set_ylabel("$\Delta R_{\mathrm{eff}}$, $t_f$–$t_0$ (nm)", fontsize=18)

    # CLEAN INWARD TICKS
    ax.tick_params(axis='both', direction='in', bottom=True, left=True, top=False, right=False, labelsize=12, width=1.0, length=5)
    ax.set_box_aspect(1)

    save_square_plot(fig, os.path.join(OUTPUT_DIR, save_filename))
    plt.show()
    plt.close(fig)

# =============================================================================
# MASTER EXECUTION PIPELINE
# =============================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- TASK 1: FIGURE A (PL ONSET CONNECTED LINE PLOT) ---
    print("=== TASK 1: Processing PL Emission Onset Files (.docx) ===")
    df_detections = batch_process_docx_files()

    if not df_detections.empty:
        plot_pl_onset_connected_line(df_detections, save_filename="fig_a_pl_onset_line_dots.png")

    # --- TASK 2 & 3: RATE CHARTS & FIGURE C ---
    print("\n=== TASK 2 & 3: Processing Master Kinetic Excel Data ===")
    try:
        xls = pd.ExcelFile(MASTER_EXCEL_PATH, engine='openpyxl')
        df_rates = pd.read_excel(xls, sheet_name=RATE_SHEET)
        df_radius = pd.read_excel(xls, sheet_name=RADIUS_SHEET)

        # Generate t0 initial & paired rate bar charts
        generate_growth_rate_charts(df_rates)

        # Generate Figure c Rate-Extent Tradeoff scatter plot (using Brus Reff data)
        plot_rate_extent_tradeoff(df_radius, save_filename="fig_c_rate_extent_tradeoff.png")

    except Exception as e:
        print(f"[!] Failed to process Master Excel data: {e}")

    print("\n✓ Pipeline execution complete. All figures saved to:", OUTPUT_DIR)
