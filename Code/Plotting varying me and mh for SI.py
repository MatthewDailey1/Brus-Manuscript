import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import re

# --- Global Font Settings ---
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial'
plt.rcParams['mathtext.bf'] = 'Arial'

# --- Configuration ---
FILE_PATH = r"C:\Users\matth\OneDrive\Desktop\Manuscript data\Brus manuscript sensitivity testing pristine MAPbI3 (1).xlsx"
SAVE_DIR = r"C:\Users\matth\Downloads"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def clean_title(title_str):
    """
    Replaces trailing variable names in the title string with MathText subscripts.
    """
    title_lower = title_str.lower()
    mapping = [
        ('particle de', r'$\kappa_{\mathrm{particle}}$'),
        ('particle d', r'$\kappa_{\mathrm{particle}}$'),
        ('env de', r'$\kappa_{\mathrm{env}}$'),
        ('me', r'$m_e$'),
        ('mh', r'$m_h$')
    ]
    for search_term, math_fmt in mapping:
        if title_lower.endswith(search_term):
            prefix = title_str[:len(title_str) - len(search_term)].strip()
            return f"{prefix} {math_fmt}"
    return title_str

def clean_label(label):
    """
    Cleans long string column names and maps them to subscripted formats.
    """
    label_str = str(label)

    var_mapping = {
        'env de': r'$\kappa_{\mathrm{env}}$',
        'particle de': r'$\kappa_{\mathrm{particle}}$',
        'particle d': r'$\kappa_{\mathrm{particle}}$',
        'me': r'$m_e$',
        'mh': r'$m_h$'
    }

    if '=' in label_str:
        left_side = label_str.split('=')[0].strip()
        right_side = label_str.split('=')[1].strip()
        left_lower = left_side.lower()

        for search_term, display_name in var_mapping.items():
            if left_lower.endswith(search_term):
                return f"{display_name} = {right_side}"

        return f"{left_side.split()[-1]} = {right_side}"

    return label_str

def get_style_map(columns):
    """
    Parses column names, extracts numeric values, identifies min/mid/max,
    and returns a mapping dictionary of column_name -> (linestyle or dash-tuple).
    Middle = Solid ('-')
    Lowest = Short dashes ((0, (3, 2))) -> 3pt line, 2pt space
    Highest = Short dots (':')
    """
    parsed = []
    for col in columns:
        match = re.search(r"=\s*([0-9.]+)", str(col))
        val = float(match.group(1)) if match else 0.0
        parsed.append((col, val))

    parsed_sorted = sorted(parsed, key=lambda x: x[1])

    # Custom short-dash tuple: (offset, (on_pt, off_pt))
    short_dash = (0, (3, 2))

    style_map = {}
    if len(parsed_sorted) == 3:
        style_map[parsed_sorted[0][0]] = short_dash  # Smallest -> short dashes
        style_map[parsed_sorted[1][0]] = '-'         # Middle -> solid
        style_map[parsed_sorted[2][0]] = ':'         # Largest -> dots
    else:
        for i, (col, _) in enumerate(parsed_sorted):
            styles = [short_dash, '-', ':']
            style_map[col] = styles[i % len(styles)]

    return style_map

def format_show_and_save(fig, ax, filename, x_limit=None, expand_y=False):
    """Helper function to format, show, and save the plot."""
    if x_limit:
        ax.set_xlim(left=0, right=x_limit)

    if expand_y:
        ax.set_ymargin(0.15)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    ax.tick_params(
        axis='both',
        direction='in',
        top=True,
        right=True,
        bottom=True,
        left=True,
        labelsize=14
    )

    ax.set_box_aspect(1)
    ax.legend(frameon=False, loc='best', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, transparent=True, dpi=300)

    plt.show()
    plt.close(fig)

def plot_radii_sheet(df, sheet_name):
    """Handles the 'Radii Changing' sheets (Columns A vs B, C, D)."""
    fig, ax = plt.subplots(figsize=(6, 6))

    time_col = df.iloc[:, 0]
    data_cols = df.iloc[:, 1:4]

    style_map = get_style_map(data_cols.columns)

    # Add soft background shading between min and max bounds
    y_min = data_cols.min(axis=1)
    y_max = data_cols.max(axis=1)
    ax.fill_between(time_col, y_min, y_max, color='gray', alpha=0.35, zorder=1)

    for col_name in data_cols.columns:
        ax.plot(
            time_col,
            data_cols[col_name],
            label=clean_label(col_name),
            color='black',
            linestyle=style_map[col_name],
            linewidth=2.5,
            zorder=2
        )

    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_ylabel("Characteristic Length (nm)", fontsize=16)
    ax.set_title(clean_title(sheet_name), fontsize=16)

    format_show_and_save(fig, ax, f"{sheet_name.replace(' ', '_')}.png", x_limit=275, expand_y=False)

def plot_growth_rate_sheet(df, sheet_name):
    """Handles the 'Growth Rate Changing' sheets (4 separate graphs)."""

    time_1 = df.iloc[:, 0]
    data_1 = df.iloc[:, 1:4]

    time_2 = df.iloc[:, 6]
    data_2 = df.iloc[:, 7:10]

    time_3 = df.iloc[:, 12]
    data_3 = df.iloc[:, 13:16]

    style_map_1 = get_style_map(data_1.columns)
    style_map_2 = get_style_map(data_2.columns)
    style_map_3 = get_style_map(data_3.columns)

    formatted_base_title = clean_title(sheet_name)

    # Graph 1: Region 1
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.fill_between(time_1, data_1.min(axis=1), data_1.max(axis=1), color='gray', alpha=0.35, zorder=1)
    for col in data_1.columns:
        ax1.plot(time_1, data_1[col], label=clean_label(col), color='black', linestyle=style_map_1[col], linewidth=2.5, zorder=2)
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_ylabel("d(Radius)/dt (nm/s)", fontsize=16)
    ax1.set_title(f"{formatted_base_title} - Region 1", fontsize=16)
    format_show_and_save(fig1, ax1, f"{sheet_name.replace(' ', '_')}_Region1.png", expand_y=True)

    # Graph 2: Region 2
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.fill_between(time_2, data_2.min(axis=1), data_2.max(axis=1), color='gray', alpha=0.35, zorder=1)
    for col in data_2.columns:
        ax2.plot(time_2, data_2[col], label=clean_label(col), color='black', linestyle=style_map_2[col], linewidth=2.5, zorder=2)
    ax2.set_xlabel("Time (s)", fontsize=16)
    ax2.set_ylabel("d(Radius)/dt (nm/s)", fontsize=16)
    ax2.set_title(f"{formatted_base_title} - Region 2", fontsize=16)
    format_show_and_save(fig2, ax2, f"{sheet_name.replace(' ', '_')}_Region2.png", expand_y=True)

    # Graph 3: Region 3
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.fill_between(time_3, data_3.min(axis=1), data_3.max(axis=1), color='gray', alpha=0.35, zorder=1)
    for col in data_3.columns:
        ax3.plot(time_3, data_3[col], label=clean_label(col), color='black', linestyle=style_map_3[col], linewidth=2.5, zorder=2)
    ax3.set_xlabel("Time (s)", fontsize=16)
    ax3.set_ylabel("d(Radius)/dt (nm/s)", fontsize=16)
    ax3.set_title(f"{formatted_base_title} - Region 3", fontsize=16)
    format_show_and_save(fig3, ax3, f"{sheet_name.replace(' ', '_')}_Region3.png", x_limit=275, expand_y=True)

    # Graph 4: Combined Regions 1, 2, and 3
    fig4, ax4 = plt.subplots(figsize=(6, 6))

    ax4.fill_between(time_1, data_1.min(axis=1), data_1.max(axis=1), color='gray', alpha=0.35, zorder=1)
    ax4.fill_between(time_2, data_2.min(axis=1), data_2.max(axis=1), color='gray', alpha=0.35, zorder=1)
    ax4.fill_between(time_3, data_3.min(axis=1), data_3.max(axis=1), color='gray', alpha=0.35, zorder=1)

    for i in range(len(data_1.columns)):
        col_1 = data_1.columns[i]
        col_2 = data_2.columns[i]
        col_3 = data_3.columns[i]

        ax4.plot(time_1, data_1.iloc[:, i], label=clean_label(col_1), color='black', linestyle=style_map_1[col_1], linewidth=2.5, zorder=2)
        ax4.plot(time_2, data_2.iloc[:, i], color='black', linestyle=style_map_2[col_2], linewidth=2.5, zorder=2)
        ax4.plot(time_3, data_3.iloc[:, i], color='black', linestyle=style_map_3[col_3], linewidth=2.5, zorder=2)

    ax4.set_xlabel("Time (s)", fontsize=16)
    ax4.set_ylabel("d(Radius)/dt (nm/s)", fontsize=16)
    ax4.set_title(f"{formatted_base_title} - Combined Regions", fontsize=16)
    format_show_and_save(fig4, ax4, f"{sheet_name.replace(' ', '_')}_Combined.png", x_limit=275, expand_y=True)

# --- Main Execution ---

# 1. Radii Changing me
df1 = pd.read_excel(FILE_PATH, sheet_name="Radii Changing me")
plot_radii_sheet(df1, "Radii Changing me")

# 2. Growth Rate Changing me
df2 = pd.read_excel(FILE_PATH, sheet_name="Growth Rate Changing me")
plot_growth_rate_sheet(df2, "Growth Rate Changing me")

# 3. Radii changing mh
df3 = pd.read_excel(FILE_PATH, sheet_name="Radii changing mh")
plot_radii_sheet(df3, "Radii changing mh")

# 4. Growth Rate Changing mh
df4 = pd.read_excel(FILE_PATH, sheet_name="Growth Rate Changing mh")
plot_growth_rate_sheet(df4, "Growth Rate Changing mh")

# 5. Radii Changing particle de
df5 = pd.read_excel(FILE_PATH, sheet_name="Radii Changing particle de")
plot_radii_sheet(df5, "Radii Changing particle de")

# 6. Growth Rate Changing particle d
df6 = pd.read_excel(FILE_PATH, sheet_name="Growth Rate Changing particle d")
plot_growth_rate_sheet(df6, "Growth Rate Changing particle d")

# 7. Radii Changing Env de
df7 = pd.read_excel(FILE_PATH, sheet_name="Radii Changing Env de")
plot_radii_sheet(df7, "Radii Changing Env de")

# 8. Growth Rate Changing Env de
df8 = pd.read_excel(FILE_PATH, sheet_name="Growth Rate Changing Env de")
plot_growth_rate_sheet(df8, "Growth Rate Changing Env de")
