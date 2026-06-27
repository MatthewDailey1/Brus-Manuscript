# -*- coding: utf-8 -*-
"""
Standalone in-situ PL fitting tool -- RAW vs SMOOTHED comparison edition.
Upgraded with Native Asymmetric Pseudo-Voigt Processing & Tracking.
+ Post-fit peak position spike filtering (Option 1).
+ Parallel frame fitting via ProcessPoolExecutor (5 worker processes).

@author: Tim Kodalle (original), adapted
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import ticker
from scipy import signal
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import scipy.integrate as integrate
import tkinter as tk
from tkinter import filedialog, simpledialog
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# =============================================================================
# PARALLEL PROCESSING SETTINGS
# =============================================================================
NUM_WORKERS = 5   # Number of CPU cores to use for fitting.
                  # Raise up to os.cpu_count() if you want to use all cores.
                  # Keep at least 1 core free for the OS / GUI.

# =============================================================================
# SMOOTHING SETTINGS -- tune these after looking at your real data
# =============================================================================
MEDIAN_FILTER_SIZE = 20
MEDIAN_FILTER_OUTLIER_FACTOR = 15  # a point is an "outlier" if > this x the local median

SAVGOL_WINDOW = 8
SAVGOL_POLYORDER = 2

# =============================================================================
# POST-FIT POSITION FILTER SETTINGS -- tune these to control spike removal
# =============================================================================
POS_FILTER_R2_THRESHOLD = 0.85   # Frames with R² below this are blanked (set to NaN)
                                  # Raise toward 0.95 for stricter quality gating
POS_FILTER_SPIKE_WINDOW = 5      # Half-width of the rolling neighbourhood (in frames)
                                  # used to compute the local mean/std for z-score rejection
POS_FILTER_SPIKE_SIGMA = 2.5     # A frame is a spike if its position deviates more than
                                  # this many std-devs from neighbours. Lower = more aggressive.
POS_FILTER_INTERPOLATE = True    # If True, linearly interpolate over blanked frames.
                                  # Set False to leave them as NaN gaps in the plot/CSV.

# =============================================================================
# CORE MODEL FUNCTIONS (Upgraded to handle Asymmetry Natively)
# =============================================================================

def sum_of_Voigts(x, *params):
    """
    Asymmetric Pseudo-Voigt profile, sum of `n` peaks plus linear background.
    Splits the width parameter into left and right components based on the peak center.
    """
    x_is_scalar = np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
    if x_is_scalar:
        x_arr = np.array([x], dtype=float)
    else:
        x_arr = np.asarray(x, dtype=float)

    params = np.array(params)
    n = (len(params) - 2) // 5  # 5 parameters per peak: amp, mu, sigma_left, sigma_right, alpha

    amps        = params[:n]
    mus         = params[n:2 * n]
    sigmas_left = params[2 * n:3 * n]
    sigmas_right= params[3 * n:4 * n]
    alphas      = params[4 * n:5 * n]
    slope       = params[-2]
    intercept   = params[-1]

    result = np.zeros_like(x_arr, dtype=float)

    for i in range(n):
        mu    = mus[i]
        amp   = amps[i]
        alpha = alphas[i]
        s_l   = sigmas_left[i]
        s_r   = sigmas_right[i]

        sigma = np.where(x_arr < mu, s_l, s_r)
        sigma = np.where(sigma <= 0, 1e-6, sigma)

        gaussian   = np.exp(-np.log(2) * ((x_arr - mu) / sigma) ** 2)
        lorentzian = 1.0 / (1.0 + ((x_arr - mu) / sigma) ** 2)

        result += amp * ((1 - alpha) * gaussian + alpha * lorentzian)

    result += slope * x_arr + intercept

    if x_is_scalar:
        return result[0]
    return result


def background(x, y0, y1):
    return y0 * x + y1


def fWHM_Voigt(x, center, maxValue, params):
    """Find FWHM by root-finding the half-max crossing on either side of the peak."""
    x1 = np.linspace(x[0], center, 5001)
    x2 = np.linspace(center, x[-1], 5001)

    y1 = sum_of_Voigts(x1, *params)
    y2 = sum_of_Voigts(x2, *params)

    root1 = np.interp(maxValue / 2, y1, x1)
    root2 = np.interp(maxValue / 2, y2[::-1], x2[::-1])

    return root2 - root1


def format_seconds_to_mmss(total_seconds):
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:04.1f}"


def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def apply_smoothing(df_fit):
    """Median-filter spike removal followed by Savitzky-Golay smoothing."""
    smoothed = np.copy(df_fit)

    if MEDIAN_FILTER_SIZE and MEDIAN_FILTER_SIZE > 1:
        local_medians = median_filter(smoothed, size=(MEDIAN_FILTER_SIZE, 1))
        outlier_mask  = smoothed > (local_medians * MEDIAN_FILTER_OUTLIER_FACTOR)
        smoothed[outlier_mask] = local_medians[outlier_mask]

    if SAVGOL_WINDOW and SAVGOL_WINDOW > 1:
        smoothed = savgol_filter(smoothed, window_length=SAVGOL_WINDOW,
                                 polyorder=SAVGOL_POLYORDER, axis=0)

    return smoothed


# =============================================================================
# POST-FIT POSITION SPIKE FILTER
# =============================================================================

def filter_peak_positions(popt_arr, r2_arr, numGauss,
                           r2_threshold=POS_FILTER_R2_THRESHOLD,
                           spike_window=POS_FILTER_SPIKE_WINDOW,
                           spike_sigma=POS_FILTER_SPIKE_SIGMA,
                           interpolate=POS_FILTER_INTERPOLATE):
    """
    Remove position spikes from fitted peak tracks using two sequential gates:

    Gate 1 -- R² quality gate:
        Frames whose R² falls below `r2_threshold` are blanked (NaN).

    Gate 2 -- Rolling z-score spike detection:
        Frames whose position deviates more than `spike_sigma` std-devs from
        the local ±`spike_window` neighbourhood are blanked.

    Optionally, blanked frames are linearly interpolated from valid neighbours.
    The original popt_arr is never modified; a copy is returned.
    """
    popt_filtered = np.copy(popt_arr)

    for peak_idx in range(int(numGauss)):
        pos_col   = int(numGauss) + peak_idx
        positions = popt_filtered[:, pos_col].copy()

        # Gate 1: R² quality gate
        bad_r2 = r2_arr < r2_threshold
        positions[bad_r2] = np.nan

        # Gate 2: rolling z-score
        n_frames = len(positions)
        for j in range(n_frames):
            if np.isnan(positions[j]):
                continue
            lo = max(0, j - spike_window)
            hi = min(n_frames, j + spike_window + 1)
            neighbourhood = positions[lo:hi]
            valid = neighbourhood[~np.isnan(neighbourhood)]
            if len(valid) < 3:
                continue
            local_mean = np.nanmean(valid)
            local_std  = np.nanstd(valid)
            if local_std > 0 and abs(positions[j] - local_mean) > spike_sigma * local_std:
                positions[j] = np.nan

        # Optional interpolation over NaN gaps
        if interpolate:
            idx        = np.arange(n_frames)
            valid_mask = ~np.isnan(positions)
            if valid_mask.sum() > 2:
                positions = np.interp(idx, idx[valid_mask], positions[valid_mask])

        popt_filtered[:, pos_col] = positions

    return popt_filtered


# =============================================================================
# PER-FRAME WORKER FUNCTION  (must be module-level for multiprocessing pickling)
# =============================================================================

def _fit_single_frame(args):
    """
    Fit one frame of spectral data.  Designed to be called by a worker process.

    Parameters
    ----------
    args : tuple
        (frame_idx, spectrum_1d, df_yCut, numGauss,
         idxLowerTH, idxUpperTH,
         peakLowerTH, peakUpperTH,
         estPeakWidth, minPeakWidth, maxPeakWidth)

    Returns
    -------
    tuple : (frame_idx, spectrum_1d, popt_row, fwhm_row, area_row, asym_row, r2)
        All NaN on failure.
    """
    (frame_idx, spectrum, df_yCut, numGauss,
     idxLowerTH, idxUpperTH,
     peakLowerTH, peakUpperTH,
     estPeakWidth, minPeakWidth, maxPeakWidth) = args

    n_params    = 5 * int(numGauss) + 2
    nan_row     = np.full(n_params, np.nan)
    nan_peaks   = np.full(int(numGauss), np.nan)

    # --- Sanitise spectrum ---
    yVals = np.where(spectrum == float('inf'), 5.0, spectrum).copy()

    idx = np.argmax(yVals[:idxUpperTH[0]])
    if idx > 0:
        yVals[idx] = yVals[idx - 1]

    # --- Peak detection ---
    noise_floor_std = np.std(yVals[:5]) if len(yVals) > 5 else 1.0
    peaks = signal.find_peaks(yVals, prominence=5 * noise_floor_std)[0]
    if len(peaks) == 0:
        return (frame_idx, yVals, nan_row, nan_peaks, nan_peaks, nan_peaks, np.nan)

    # --- Build initial guesses and bounds ---
    spec_max    = max(np.max(yVals), 1.0)
    edge_points = max(1, min(15, len(yVals) // 10))
    left_bkg    = np.mean(yVals[:edge_points])
    right_bkg   = np.mean(yVals[-edge_points:])
    x_range     = df_yCut[-1] - df_yCut[0]

    estLinBkg   = (right_bkg - left_bkg) / x_range if x_range > 0 else 0.0
    estConstBkg = left_bkg - estLinBkg * df_yCut[0]

    minLinBkg   = estLinBkg  - spec_max * 0.2
    maxLinBkg   = estLinBkg  + spec_max * 0.2
    minConstBkg = estConstBkg - spec_max * 0.8
    maxConstBkg = estConstBkg + spec_max * 0.8

    estAmplitudes = []
    minAmplitudes = []
    maxAmplitudes = []
    estPositions  = []

    for ii in range(int(numGauss)):
        window_slice  = yVals[idxLowerTH[ii]:idxUpperTH[ii]]
        local_max_idx = idxLowerTH[ii] + int(np.argmax(window_slice))
        estPositions.append(df_yCut[local_max_idx])
        estAmplitudes.append(float(np.max(window_slice)))
        minAmplitudes.append(0.0)
        maxAmplitudes.append(np.inf)

    estAlphas    = [0.24]  * int(numGauss)
    minAlphas    = [0.0]   * int(numGauss)
    maxAlphas    = [1.0]   * int(numGauss)

    estWidthsLeft  = list(estPeakWidth)
    estWidthsRight = list(estPeakWidth)
    minWidthsLeft  = list(minPeakWidth)
    minWidthsRight = list(minPeakWidth)
    maxWidthsLeft  = list(maxPeakWidth)
    maxWidthsRight = list(maxPeakWidth)

    estParams   = (estAmplitudes + estPositions +
                   estWidthsLeft + estWidthsRight + estAlphas +
                   [estLinBkg, estConstBkg])
    lowerBounds = (minAmplitudes + peakLowerTH +
                   minWidthsLeft + minWidthsRight + minAlphas +
                   [minLinBkg, minConstBkg])
    upperBounds = (maxAmplitudes + peakUpperTH +
                   maxWidthsLeft + maxWidthsRight + maxAlphas +
                   [maxLinBkg, maxConstBkg])

    # --- Curve fit ---
    try:
        popt_row, _ = curve_fit(
            sum_of_Voigts, df_yCut, yVals,
            p0=estParams, bounds=(lowerBounds, upperBounds)
        )
    except Exception:
        return (frame_idx, yVals, nan_row, nan_peaks, nan_peaks, nan_peaks, np.nan)

    # --- Derived peak metrics ---
    fwhm_row = np.full(int(numGauss), np.nan)
    area_row = np.full(int(numGauss), np.nan)
    asym_row = np.full(int(numGauss), np.nan)

    for ii in range(int(numGauss)):
        amp_val   = popt_row[ii]
        mu_val    = popt_row[int(numGauss) + ii]
        sig_l_val = popt_row[2 * int(numGauss) + ii]
        sig_r_val = popt_row[3 * int(numGauss) + ii]
        alpha_val = popt_row[4 * int(numGauss) + ii]

        parameters  = [amp_val, mu_val, sig_l_val, sig_r_val, alpha_val, 0, 0]
        peak_center = mu_val

        max_value     = float(np.asarray(sum_of_Voigts(peak_center, *parameters)).flat[0])
        fwhm_row[ii]  = fWHM_Voigt(df_yCut, peak_center, max_value, parameters)
        asym_row[ii]  = sig_r_val / sig_l_val if sig_l_val > 0 else np.nan
        area_row[ii]  = integrate.quad(
            lambda x: float(np.asarray(sum_of_Voigts(x, *parameters)).flat[0]),
            -np.inf, np.inf
        )[0]

    # --- R² ---
    try:
        fit_y = sum_of_Voigts(df_yCut, *popt_row)
        r2    = calculate_r2(yVals, fit_y)
    except Exception:
        r2 = np.nan

    return (frame_idx, yVals, popt_row, fwhm_row, area_row, asym_row, r2)


# =============================================================================
# PARALLEL SINGLE-PASS FITTING
# =============================================================================

def fit_dataset(df_yCut, df_xCutFit, df_fit, numGauss,
                peakLowerTH, peakUpperTH, estPeakWidth, minPeakWidth, maxPeakWidth,
                desc='Fitting'):
    """
    Fit every frame in df_fit using NUM_WORKERS parallel processes.

    Each frame is an independent curve_fit call, so there is no shared state
    between workers — perfect for ProcessPoolExecutor.

    The result arrays are assembled in frame-index order after all workers finish.
    """
    n_frames  = np.shape(df_fit)[1]
    n_params  = 5 * int(numGauss) + 2

    # Pre-compute energy-index bounds for each peak (same for every frame)
    idxLowerTH = [0] * int(numGauss)
    idxUpperTH = [0] * int(numGauss)
    for i in range(int(numGauss)):
        idxLowerTH[i] = next(
            (k for k, v in enumerate(df_yCut) if v > peakLowerTH[i]), 0)
        idxUpperTH[i] = next(
            (k for k, v in enumerate(df_yCut) if v > peakUpperTH[i]), len(df_yCut) - 1)

    # Build argument list: one tuple per frame
    job_args = [
        (i, df_fit[:, i].copy(), df_yCut,
         numGauss, idxLowerTH, idxUpperTH,
         list(peakLowerTH), list(peakUpperTH),
         list(estPeakWidth), list(minPeakWidth), list(maxPeakWidth))
        for i in range(n_frames)
    ]

    # Allocate result arrays
    yVals       = np.copy(df_fit)
    popt        = np.full((n_frames, n_params),       np.nan)
    peakFWHM    = np.full((n_frames, int(numGauss)),  np.nan)
    peakArea    = np.full((n_frames, int(numGauss)),  np.nan)
    peakAsymmetry = np.full((n_frames, int(numGauss)),np.nan)
    r2_values   = np.full(n_frames, np.nan)

    # Submit all frames to the process pool and collect as they complete
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_fit_single_frame, arg): arg[0] for arg in job_args}
        with tqdm(total=n_frames, desc=desc) as pbar:
            for future in as_completed(futures):
                frame_idx, yVals_i, popt_row, fwhm_row, area_row, asym_row, r2 = future.result()
                yVals[:, frame_idx]        = yVals_i
                popt[frame_idx, :]         = popt_row
                peakFWHM[frame_idx, :]     = fwhm_row
                peakArea[frame_idx, :]     = area_row
                peakAsymmetry[frame_idx, :]= asym_row
                r2_values[frame_idx]       = r2
                pbar.update(1)

    return yVals, popt, peakFWHM, peakArea, peakAsymmetry, r2_values


# =============================================================================
# DUAL-PASS WRAPPER: Fits raw and smoothed data, displays, saves to CSV
# =============================================================================

def plFitting(df_yCut, df_xCutFit, df_fit, show_every, numGauss,
              peakLowerTH, peakUpperTH, estPeakWidth, minPeakWidth, maxPeakWidth,
              name_d, name):

    df_fit_smoothed = apply_smoothing(df_fit)

    print(f"Fitting RAW data across {NUM_WORKERS} workers...")
    yVals_raw, popt_raw, fwhm_raw, area_raw, asym_raw, r2_raw = fit_dataset(
        df_yCut, df_xCutFit, df_fit, numGauss,
        list(peakLowerTH), list(peakUpperTH), list(estPeakWidth),
        list(minPeakWidth), list(maxPeakWidth), desc='Fitting RAW spectra (eV)'
    )

    print(f"Fitting SMOOTHED data across {NUM_WORKERS} workers...")
    yVals_smooth, popt_smooth, fwhm_smooth, area_smooth, asym_smooth, r2_smooth = fit_dataset(
        df_yCut, df_xCutFit, df_fit_smoothed, numGauss,
        list(peakLowerTH), list(peakUpperTH), list(estPeakWidth),
        list(minPeakWidth), list(maxPeakWidth), desc='Fitting SMOOTHED spectra (eV)'
    )

    # --- Apply post-fit position spike filtering ---
    print("Applying post-fit position spike filter...")
    popt_raw_filtered    = filter_peak_positions(popt_raw,    r2_raw,    numGauss)
    popt_smooth_filtered = filter_peak_positions(popt_smooth, r2_smooth, numGauss)

    frames = range(0, len(df_xCutFit))

    # --- SIDE-BY-SIDE INTERACTIVE SLIDER REVIEW ---
    fig, (ax_raw, ax_smooth) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.22)

    def safe_fit(popt_arr, frame_idx):
        if np.isnan(popt_arr[frame_idx, 0]):
            return np.full_like(df_yCut, np.nan)
        return sum_of_Voigts(df_yCut, *popt_arr[frame_idx, :])

    raw_data_line,  = ax_raw.plot(df_yCut, yVals_raw[:, 0],   'ob', label='Raw data',      alpha=0.6, markersize=3)
    raw_fit_line,   = ax_raw.plot(df_yCut, safe_fit(popt_raw, 0), '-r', lw=2, label='Fit')
    ax_raw.set_xlim(np.min(df_yCut), np.max(df_yCut))
    ax_raw.set_xlabel("Photon Energy (eV)")
    ax_raw.set_ylabel("Intensity (a.u.)")
    ax_raw.set_title("Raw")
    ax_raw.legend(loc='best')

    smooth_data_line, = ax_smooth.plot(df_yCut, yVals_smooth[:, 0], 'og', label='Smoothed data', alpha=0.6, markersize=3)
    smooth_fit_line,  = ax_smooth.plot(df_yCut, safe_fit(popt_smooth, 0), '-r', lw=2, label='Fit')
    ax_smooth.set_xlim(np.min(df_yCut), np.max(df_yCut))
    ax_smooth.set_xlabel("Photon Energy (eV)")
    ax_smooth.set_ylabel("Intensity (a.u.)")
    ax_smooth.set_title("Smoothed (median + Savitzky-Golay)")
    ax_smooth.legend(loc='best')

    ax_slider = plt.axes([0.15, 0.07, 0.7, 0.03])
    slider    = Slider(ax_slider, 'Frame', 0, len(frames) - 1, valinit=0, valfmt='%d')

    def update_plot(val):
        frame_idx = int(slider.val)

        new_y_raw   = yVals_raw[:, frame_idx]
        new_fit_raw = safe_fit(popt_raw, frame_idx)
        raw_data_line.set_ydata(new_y_raw)
        raw_fit_line.set_ydata(new_fit_raw)

        new_y_smooth   = yVals_smooth[:, frame_idx]
        new_fit_smooth = safe_fit(popt_smooth, frame_idx)
        smooth_data_line.set_ydata(new_y_smooth)
        smooth_fit_line.set_ydata(new_fit_smooth)

        mmss_time      = format_seconds_to_mmss(df_xCutFit[frame_idx])
        r2_raw_text    = f"{r2_raw[frame_idx]:.4f}"    if not np.isnan(r2_raw[frame_idx])    else "N/A"
        r2_smooth_text = f"{r2_smooth[frame_idx]:.4f}" if not np.isnan(r2_smooth[frame_idx]) else "N/A"

        finite_raw = new_fit_raw[~np.isnan(new_fit_raw)]
        if finite_raw.size > 0:
            ax_raw.set_ylim(min(np.min(new_y_raw), np.min(finite_raw)) * 0.95,
                            max(np.max(new_y_raw), np.max(finite_raw)) * 1.05)
        else:
            ax_raw.set_ylim(np.min(new_y_raw) * 0.9, np.max(new_y_raw) * 1.1)

        finite_smooth = new_fit_smooth[~np.isnan(new_fit_smooth)]
        if finite_smooth.size > 0:
            ax_smooth.set_ylim(min(np.min(new_y_smooth), np.min(finite_smooth)) * 0.95,
                                max(np.max(new_y_smooth), np.max(finite_smooth)) * 1.05)
        else:
            ax_smooth.set_ylim(np.min(new_y_smooth) * 0.9, np.max(new_y_smooth) * 1.1)

        ax_raw.set_title(f"Raw | R\u00b2: {r2_raw_text}")
        ax_smooth.set_title(f"Smoothed | R\u00b2: {r2_smooth_text}")
        fig.suptitle(f"Frame: {frame_idx} (Elapsed Time: {mmss_time})")
        fig.canvas.draw_idle()

    slider.on_changed(update_plot)
    update_plot(0)
    plt.show(block=True)

    # --- Time-evolution summary plots: raw vs smoothed overlay per peak ---
    for i in range(0, int(numGauss)):
        fig2, ax1 = plt.subplots(figsize=(7, 5))

        ax1.plot(df_xCutFit, popt_raw_filtered[:, int(numGauss) + i],
                 'o-', label='Position (raw, filtered)', alpha=0.6)
        ax1.plot(df_xCutFit, popt_smooth_filtered[:, int(numGauss) + i],
                 's-', label='Position (smoothed, filtered)', alpha=0.8)
        ax1.plot(df_xCutFit, popt_raw[:, int(numGauss) + i],
                 ':', color='steelblue', alpha=0.25, label='Position (raw, unfiltered)')
        ax1.plot(df_xCutFit, popt_smooth[:, int(numGauss) + i],
                 ':', color='darkorange', alpha=0.25, label='Position (smoothed, unfiltered)')

        ax2 = ax1.twinx()
        ax2.plot(df_xCutFit, popt_raw[:, i],    'g^--', label='Intensity (raw)',      alpha=0.5)
        ax2.plot(df_xCutFit, popt_smooth[:, i], 'gv--', label='Intensity (smoothed)', alpha=0.8)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('PL Position (eV)')
        ax2.set_ylabel('PL Intensity (a.u.)')
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))
        fig2.suptitle(f'Fit Results Peak {i + 1} {name_d} (Raw vs Smoothed, filtered)', fontsize=13)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)

    if int(numGauss) > 0:
        plt.show(block=True)

    # --- SAVE RESULTS ---
    def build_results_df(popt_arr, fwhm_arr, area_arr, asym_arr, r2_arr):
        dfPeaks = pd.DataFrame()
        dfPeaks['Fit-Time_' + name_d] = df_xCutFit
        for i in range(0, int(numGauss)):
            data = np.array([
                area_arr[:, i],
                popt_arr[:, i],
                popt_arr[:, int(numGauss) + i],
                fwhm_arr[:, i],
                popt_arr[:, 2 * int(numGauss) + i],
                popt_arr[:, 3 * int(numGauss) + i],
                asym_arr[:, i],
                popt_arr[:, 4 * int(numGauss) + i]
            ])
            cols = [
                f'Peak{i+1}Area_{name_d}',       f'Peak{i+1}Amplitude_{name_d}',
                f'Peak{i+1}Pos_{name_d}',         f'Peak{i+1}FWHM_{name_d}',
                f'Peak{i+1}SigmaLeft_{name_d}',   f'Peak{i+1}SigmaRight_{name_d}',
                f'Peak{i+1}AsymmetryFactor_{name_d}', f'Peak{i+1}Alpha_{name_d}'
            ]
            dfPeaks = pd.concat([dfPeaks, pd.DataFrame(data.T, columns=cols)], axis=1)
        dfPeaks[f'R2_{name_d}'] = r2_arr
        return dfPeaks.fillna('nan')

    def save_csv(df, base_filename):
        output_path = os.path.join(name, base_filename)
        counter = 1
        while True:
            try:
                df.to_csv(output_path, index=False)
                print(f"Successfully saved results to: {output_path}")
                return output_path
            except PermissionError:
                stem, ext = os.path.splitext(base_filename)
                output_path = os.path.join(name, f'{stem}_{counter}{ext}')
                counter += 1

    save_csv(build_results_df(popt_raw,            fwhm_raw,    area_raw,    asym_raw,    r2_raw),
             f'PL_FitResults_{name_d}_raw.csv')
    save_csv(build_results_df(popt_smooth,          fwhm_smooth, area_smooth, asym_smooth, r2_smooth),
             f'PL_FitResults_{name_d}_smoothed.csv')
    save_csv(build_results_df(popt_raw_filtered,    fwhm_raw,    area_raw,    asym_raw,    r2_raw),
             f'PL_FitResults_{name_d}_raw_filtered.csv')
    save_csv(build_results_df(popt_smooth_filtered, fwhm_smooth, area_smooth, asym_smooth, r2_smooth),
             f'PL_FitResults_{name_d}_smoothed_filtered.csv')

    return (popt_raw, fwhm_raw, area_raw, asym_raw, r2_raw), \
           (popt_smooth, fwhm_smooth, area_smooth, asym_smooth, r2_smooth)


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    # NOTE: The `if __name__ == "__main__":` guard is REQUIRED on Windows/macOS
    # when using multiprocessing.  Without it, spawned worker processes would
    # re-execute the top-level script and crash.  Never remove this guard.

    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder with PL TXT Files")
    if not folder_path:
        exit()

    current_folder   = os.path.basename(os.path.normpath(folder_path))
    parent_folder    = os.path.basename(os.path.dirname(os.path.normpath(folder_path)))
    sample_name_header = f"{parent_folder}_{current_folder}"

    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', var)])

    sample_data  = pd.read_csv(os.path.join(folder_path, files[0]), sep=r'\s+', engine='c', header=None)
    energy_col   = pd.to_numeric(sample_data[0], errors='coerce').dropna()
    df_yCut      = energy_col.values
    valid_indices = energy_col.index

    intensities_list, raw_timestamps = [], []

    print("Reading files and modification timestamps...")
    for file in files:
        full_file_path = os.path.join(folder_path, file)
        try:
            data_frame = pd.read_csv(full_file_path, sep=r'\s+', engine='c', header=None)
            intensities_list.append(pd.to_numeric(data_frame[1], errors='coerce').iloc[valid_indices].values)
            raw_timestamps.append(os.path.getmtime(full_file_path))
        except Exception as e:
            print(f"Skipping file {file}: {e}")

    df_fit      = np.array(intensities_list).T
    base_time   = raw_timestamps[0]
    df_xCutFit  = np.array([t - base_time for t in raw_timestamps])
    df_yCut     = 1239.84193 / df_yCut  # nm -> eV

    num_peaks  = simpledialog.askinteger("Input", "How many peaks?", initialvalue=1)
    show_every = simpledialog.askinteger("Input", "Plot every 'N' frames (unused by slider, kept for compatibility):", initialvalue=10)

    min_fit_window = 1.35  # eV
    max_fit_window = 1.95  # eV

    window_mask = (df_yCut >= min_fit_window) & (df_yCut <= max_fit_window)
    df_yCut     = df_yCut[window_mask]
    df_fit      = df_fit[window_mask, :]

    sort_idx = np.argsort(df_yCut)
    df_yCut  = df_yCut[sort_idx]
    df_fit   = df_fit[sort_idx, :]

    forced_lower_th = [1.35]  * num_peaks
    forced_upper_th = [1.90]  * num_peaks
    est_peak_width  = [0.01]  * num_peaks
    forced_min_width= [0.0005]* num_peaks
    forced_max_width= [0.25]  * num_peaks

    output_dir = folder_path

    plFitting(df_yCut, df_xCutFit, df_fit, show_every, num_peaks,
              forced_lower_th, forced_upper_th, est_peak_width,
              forced_min_width, forced_max_width, sample_name_header, output_dir)

    print("\nProcessing complete! Peaks tracked and exported in eV.")
