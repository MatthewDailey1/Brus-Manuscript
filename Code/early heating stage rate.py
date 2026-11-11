# -*- coding: utf-8 -*-
"""
Integrated Nanoparticle Kinetic & Onset Analysis Pipeline
=============================================================================
1. Parses PL fit reports (.docx) to generate PL Emission Onset charts.
2. Extracts Master Excel growth data to generate initial (t0) and paired
   side-by-side (t0 vs tf) heating-stage rate figures.
All figures are exported as pixel-perfect, publication-ready square plots.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
from docx import Document

# Global typography and aesthetic setup
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['savefig.transparent'] = True

# =============================================================================
# GLOBAL EXPERIMENTAL CONFIGURATION
# =============================================================================
OUTPUT_DIR = r"C:\Users\matth\Downloads"
MASTER_FILE_PATH = r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\260717_Master_Data.xlsx"
TARGET_SHEET = 'GrowthRate_All'

PL_SAMPLES = [
    {
        "label": "Control",
        "header_key": "control",
        "docx_path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\Reports\260717_PL_FitResults_control.docx",
        "color": "#232323"  # Deep Charcoal / Black
    },
    {
        "label": "3-APA",
        "header_key": "3APA",
        "docx_path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\Reports\260717_PL_FitResults_3APA.docx",
        "color": "#E07A5F"  # Coral / Terracotta
    },
    {
        "label": "4-ABA",
        "header_key": "4ABA",
        "docx_path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\Reports\260717_PL_FitResults_4ABA.docx",
        "color": "#38B000"  # Kelly Green
    },
    {
        "label": "5-AVA",
        "header_key": "5AVA",
        "docx_path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\Reports\260717_PL_FitResults_5AVA.docx",
        "color": "#A381C6"  # Muted Purple
    },
    {
        "label": "7-AHA",
        "header_key": "7AHA",
        "docx_path": r"C:\Users\matth\OneDrive\Desktop\Manuscript data\260717_Data\Reports\260717_PL_FitResults_7AHA.docx",
        "color": "#70E4BC"  # Mint / Soft Teal
    }
]

# Exact sample-specific time bounds for growth rates (seconds)
SAMPLE_TIME_BOUNDS = {
    "Control": {"t0": 83.0, "tf": 99.0},
    "3-APA":   {"t0": 83.0, "tf": 101.0},
    "4-ABA":   {"t0": 84.0, "tf": 100.0},
    "5-AVA":   {"t0": 84.0, "tf": 100.0},
    "7-AHA":   {"t0": 83.0, "tf": 100.0}
}

# =============================================================================
# DATA EXTRACTION HELPERS (.DOCX & EXCEL)
# =============================================================================
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
            results.append(bounds)
            print(f"  ✓ Parsed {cfg['label']}: Onset @ {bounds['r1_start']}s")
        except Exception as e:
            print(f"  [!] Could not parse .docx for {cfg['label']} ({cfg['docx_path']}): {e}")

    return pd.DataFrame(results)

def get_sample_xy_data(df_master, header_key):
    """Extracts, cleans, and sorts all X and Y data for a given sample from Excel."""
    sample_cols = [col for col in df_master.columns if header_key in str(col)]
    if len(sample_cols) < 6:
        return np.array([]), np.array([])

    df_sample = df_master[sample_cols]

    x_list, y_list = [], []
    for i in range(0, 6, 2):
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

def extract_rate_at_exact_time(x_arr, y_arr, target_time):
    """Interpolates kinetic rate at a specific timestamp."""
    if len(x_arr) == 0: return 0.0
    return np.interp(target_time, x_arr, y_arr)

# =============================================================================
# PLOTTING ROUTINES
# =============================================================================
def plot_region_stacked_chart(df_results, with_outline=True, save_filename="pl_startup_timeline_square.png"):
    """Generates PL onset timeline square chart using strict bounding box transformation."""
    fig = plt.figure(figsize=(6.0, 6.0), dpi=300)
    fig.patch.set_alpha(0.0)

    ax = fig.add_axes([0.13, 0.13, 0.82, 0.82])
    ax.set_facecolor('none')
    ax.set_box_aspect(1.0)

    labels = df_results['label']
    colors = df_results['color']
    width = 0.55
    x_coords = np.arange(len(labels))
    heights = df_results['r1_start']

    edge_color = 'black' if with_outline else 'none'
    line_width = 1.2 if with_outline else 0.0

    ax.bar(
        x_coords, heights, width,
        color=colors, edgecolor=edge_color, linewidth=line_width
    )

    ax.set_xticks(x_coords)
    ax.set_xticklabels(labels, fontsize=14, fontname='Arial')
    ax.set_ylabel('Time before PL Detected (s)', fontsize=14, fontname='Arial', labelpad=6)
    ax.set_ylim(0, heights.max() + 5.0)

    ax.tick_params(axis='x', bottom=False, top=False)
    ax.tick_params(axis='y', direction='in', right=True, width=0.8, length=5, labelsize=14)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Calculate exact 1:1 pixel square crop
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_bbox = fig.get_tightbbox(renderer).padded(0.04)

    content_w, content_h = tight_bbox.width, tight_bbox.height
    side = max(content_w, content_h)

    x0 = tight_bbox.x0 - (side - content_w) / 2.0
    y0 = tight_bbox.y0 - (side - content_h) / 2.0
    square_bbox = Bbox.from_bounds(x0, y0, side, side)

    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path, dpi=300, transparent=True, bbox_inches=square_bbox)
    plt.show()
    plt.close(fig)

def generate_growth_rate_charts(df_master):
    """Generates initial rate (t0) and side-by-side grouped (t0 vs tf) rate charts."""
    sample_data = {}
    for cfg in PL_SAMPLES:
        x, y = get_sample_xy_data(df_master, cfg['header_key'])
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

        early_rates.append(extract_rate_at_exact_time(x, y, target_time=t0))
        late_rates.append(extract_rate_at_exact_time(x, y, target_time=tf))

    # --- CHART 1: INITIAL RATE (t0) ---
    print("--> Generating Square Initial Growth Rate Graph (t0)...")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

    ax.bar(x_coords, early_rates, width=0.6, color=colors, edgecolor='none')
    ax.set_xticks(x_coords)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_ylabel("Initial Growth Rate d(Radius)/dt (nm s$^{-1}$)", fontsize=14)
    ax.set_box_aspect(1)

    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pub_square_t0_rates.png"), dpi=300)
    plt.show()
    plt.close(fig)

    # --- CHART 2: GROUPED SIDE-BY-SIDE (t0 vs tf) ---
    print("--> Generating Square Side-by-Side Growth Rate Graph (t0 vs tf)...")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

    # Left Bar: t0 (Solid fill, borderless)
    ax.bar(x_coords - bar_offset, early_rates, bar_width,
           color=colors, edgecolor='none')

    # Right Bar: tf (Hatched pattern '//', borderless frame)
    ax.bar(x_coords + bar_offset, late_rates, bar_width,
           color=colors, alpha=0.6, hatch='//', edgecolor='white', linewidth=0)

    ax.set_xticks(x_coords)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_ylabel("d(Radius)/dt (nm s$^{-1}$)", fontsize=14)
    ax.set_box_aspect(1)

    # Custom temperature legend
    t0_patch = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=1.0,
                               label='$t_0$ @ 25 °C')
    tf_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', linewidth=1.0,
                               label='$t_f$ @ 100 °C')

    ax.legend(handles=[t0_patch, tf_patch], frameon=False, fontsize=12)

    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pub_square_side_by_side_rates.png"), dpi=300)
    plt.show()
    plt.close(fig)

# =============================================================================
# MASTER EXECUTION PIPELINE
# =============================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- STEP 1: PL EMISSION ONSET TIMELINES ---
    print("=== TASK 1: Processing PL Emission Onset Files (.docx) ===")
    df_detections = batch_process_docx_files()

    if not df_detections.empty:
        print("  Generating PL Onset Chart (Outlined)...")
        plot_region_stacked_chart(
            df_detections, with_outline=True,
            save_filename="pl_startup_timeline_square_outlined.png"
        )

        print("  Generating PL Onset Chart (No Outline)...")
        plot_region_stacked_chart(
            df_detections, with_outline=False,
            save_filename="pl_startup_timeline_square_no_outline.png"
        )

    # --- STEP 2 & 3: HEATING-STAGE KINETIC RATE CHARTS ---
    print("\n=== TASK 2 & 3: Processing Master Kinetic Excel Data ===")
    try:
        df_master = pd.read_excel(MASTER_FILE_PATH, sheet_name=TARGET_SHEET, engine='openpyxl')
        generate_growth_rate_charts(df_master)
    except Exception as e:
        print(f"[!] Failed to process Master Excel data: {e}")

    print("\n✓ Pipeline execution complete. All square figures saved to:", OUTPUT_DIR)
