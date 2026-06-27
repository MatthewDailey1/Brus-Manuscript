import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import fsolve

# Clean classic/scientific style base
plt.style.use('seaborn-v0_8-whitegrid')

# =========================================================================
# MATERIAL PHYSICAL PARAMETERS (MAPbI3 Perovskite Constants)
# =========================================================================
E_BULK = 1.60          # Bulk bandgap of MAPbI3 at room temp (eV)
M_E = 0.15 * 9.11e-31  # Effective mass of electron (kg)
M_H = 0.25 * 9.11e-31  # Effective mass of hole (kg)
REDUCED_MASS = 1.0 / (1.0/M_E + 1.0/M_H)

H_BAR = 1.054e-34      # Reduced Planck constant (J*s)
E_CHARGE = 1.602e-19   # Elementary charge (C)
EPSILON_R = 6.5        # High-frequency dielectric constant for MAPbI3
EPSILON_0 = 8.854e-12  # Permittivity of free space (F/m)

def calculate_brus_diameter(e_pl_ev):
    """
    Solves the Brus Equation numerically to map PL Emission Energy (eV)
    directly to a physical Crystallite Diameter (nm).
    """
    if e_pl_ev <= E_BULK:
        return np.nan

    delta_e_joules = (e_pl_ev - E_BULK) * E_CHARGE

    def objective(R):
        if R <= 0: return 1e9
        confinement = (np.pi**2 * H_BAR**2) / (2 * REDUCED_MASS * R**2)
        coulomb = (1.786 * E_CHARGE**2) / (4 * np.pi * EPSILON_R * EPSILON_0 * R)
        return confinement - coulomb - delta_e_joules

    r_initial = 3.0e-9
    r_solved, info, ier, mesg = fsolve(objective, r_initial, full_output=True)

    if ier == 1 and r_solved[0] > 0:
        return 2.0 * r_solved[0] * 1e9
    return np.nan

# =========================================================================
# FIXED FILE PATH CONFIGURATION (CONTROL EXPLICIT DISK LOADING)
# =========================================================================
SCHERRER_CONTROL_PATH = r"G:\LBL_data\Brus_Working_data\Manuscript data\MAPI_Control_tube_S1_36_5min\processed_data\scherrer_kinetics_output.csv"
PL_MASTER_CONTROL_PATH = r"G:\LBL_data\Brus_Working_data\Manuscript data\Controls_S1_5min\260621_Data\260621_Master_Data.xlsx"

# Clarified variable names so you know exactly where they apply
PL_TARGET_SHEET = 'GrowthRate_All'

METHOD_STYLES = {
    'Scherrer': {'linestyle': '-',  'marker': 'o', 'label': 'Control (Scherrer L_c)'},
    'Brus':     {'linestyle': '--', 'marker': 's', 'label': 'Control (Brus via PL Peak)'}
}

control_dataset = {}

# =========================================================================
# DATA LOADING ENGINE
# =========================================================================

# 1. Load Control Scherrer Sizing from GIWAXS CSV (No sheets involved)
path_scherrer = Path(SCHERRER_CONTROL_PATH)
if path_scherrer.exists():
    try:
        df = pd.read_csv(path_scherrer, engine='python')
        size_col = [c for c in df.columns if 'size' in c.lower() or 'scherrer' in c.lower()]
        time_col = [c for c in df.columns if 'time' in c.lower() or 'frame' in c.lower()]

        if size_col and time_col:
            control_dataset['Scherrer'] = pd.DataFrame({
                'Time': df[time_col[0]].values,
                'Size_nm': df[size_col[0]].values
            })
            print(f"--> Successfully loaded Scherrer CSV: {path_scherrer.name}")
    except Exception as e:
        print(f"[!] Error reading Scherrer CSV: {e}")
else:
    print(f"[!] Warning: Scherrer file not found at:\n    {SCHERRER_CONTROL_PATH}")

# 2. Load Control PL Data from Master Workbook using the explicit target sheet
path_pl = Path(PL_MASTER_CONTROL_PATH)
if path_pl.exists():
    try:
        excel_file = pd.ExcelFile(path_pl, engine='openpyxl')
        if PL_TARGET_SHEET in excel_file.sheet_names:
            df_pl = pd.read_excel(excel_file, sheet_name=PL_TARGET_SHEET)

            times = df_pl.iloc[:, 0].dropna().to_numpy()
            energies = df_pl.iloc[:, 1].dropna().to_numpy()

            brus_diameters = np.array([calculate_brus_diameter(E) for E in energies])

            control_dataset['Brus'] = pd.DataFrame({
                'Time': times,
                'Size_nm': brus_diameters
            })
            print(f"--> Successfully loaded PL workbook sheet: '{PL_TARGET_SHEET}'")
        else:
            print(f"[!] Error: Target sheet '{PL_TARGET_SHEET}' missing from PL Excel file.")
    except Exception as e:
        print(f"[!] Error parsing PL Excel file: {e}")
else:
    print(f"[!] Warning: PL Workbook not found at:\n    {PL_MASTER_CONTROL_PATH}")

# =========================================================================
# ISOLATED PLOTTING ENGINE
# =========================================================================
if not control_dataset:
    sys.exit("\n[Exit] No Control datasets loaded successfully. Plotting aborted.")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
control_color = 'black'

for method_name in ['Scherrer', 'Brus']:
    if method_name in control_dataset:
        df_plot = control_dataset[method_name].dropna()
        if df_plot.empty:
            continue

        df_plot = df_plot.sort_values(by='Time')
        t_vals = df_plot['Time'].values
        s_vals = df_plot['Size_nm'].values

        m_style = METHOD_STYLES[method_name]

        ax.plot(
            t_vals,
            s_vals,
            linestyle=m_style['linestyle'],
            linewidth=2.5,
            color=control_color,
            alpha=0.85,
            label=m_style['label'],

            marker=m_style['marker'],
            markevery=20,
            markersize=5.0,
            markerfacecolor='none',
            markeredgecolor=control_color,
            markeredgewidth=1.0
        )

ax.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
ax.set_ylabel("Crystallite Size / Coherence Length (nm)", fontsize=12, fontweight='bold')
ax.set_title("Control Sizing Comparison: Brus vs. Scherrer", fontsize=13, fontweight='bold', pad=15)

ax.set_xlim(left=45)
all_computed_sizes = [
    val for method in control_dataset.values()
    for val in method['Size_nm'].values if np.isfinite(val)
]
if all_computed_sizes:
    ax.set_ylim(0, max(all_computed_sizes) * 1.15)

ax.grid(True, linestyle=':', alpha=0.5)
ax.tick_params(direction='in', top=True, right=True, labelsize=11)
ax.legend(loc='best', frameon=True, fontsize=10, facecolor='white', edgecolor='gainsboro')

plt.tight_layout()
plt.show()
