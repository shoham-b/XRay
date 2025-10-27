import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz


def find_all_peaks_naive(
    df: pd.DataFrame,
    threshold: float = 0.1,
    distance: int = 1,
    prominence: float | None = None,
    width: int | None = None,
) -> np.ndarray:
    """Finds all local maxima (spikes) in the data that meet specified criteria."""
    intensities = df["Intensity"].values
    max_intensity = intensities.max()
    height = threshold * max_intensity if threshold is not None else None
    prom = (prominence * max_intensity) if (prominence is not None) else None
    peaks, _ = find_peaks(
        intensities, height=height, distance=distance, prominence=prom, width=width
    )
    return peaks


def voigt(x, amplitude, mean, sigma, gamma):
    """Voigt profile."""
    z = (x - mean + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def bremsstrahlung_bg(x, bg_amp, x_offset, bg_scale):
    """A Maxwell-Boltzmann-like model for the Bremsstrahlung background."""
    x_shifted = x - x_offset
    x_shifted[x_shifted < 0] = 0
    return bg_amp * x_shifted * np.exp(-x_shifted / (bg_scale + 1e-9))


def fit_global_background(
    df: pd.DataFrame, initial_peaks: np.ndarray, window: int = 20
) -> tuple | None:
    """Fits a single Bremsstrahlung background to the entire spectrum."""
    x_data = df["Angle"].values
    y_data = df["Intensity"].values

    # Create a mask to exclude peak regions
    mask = np.ones_like(x_data, dtype=bool)
    for idx in initial_peaks:
        left = max(0, idx - window)
        right = min(len(x_data), idx + window + 1)
        mask[left:right] = False

    x_bg = x_data[mask]
    y_bg = y_data[mask]

    if len(x_bg) < 3:
        return None  # Not enough points to fit a background

    # Guess and fit the background
    try:
        bg_amp_guess = np.median(y_bg) if y_bg.size > 0 else 1.0
        x_offset_guess = x_data.min() - (x_data.max() - x_data.min()) * 0.1
        bg_scale_guess = (x_data.max() - x_data.min()) * 2
        guess = [bg_amp_guess, x_offset_guess, bg_scale_guess]

        lower = [0.0, -np.inf, 1e-6]
        upper = [y_data.max() * 2, x_data.max(), np.inf]

        popt, _ = curve_fit(
            bremsstrahlung_bg, x_bg, y_bg, p0=guess, bounds=(lower, upper), maxfev=10000
        )
        return popt
    except Exception:
        return None


def double_voigt(x, amp_a, mean_a, sigma, gamma, amp_b_ratio):
    """Composite model of two Voigt profiles for K-alpha and K-beta."""
    mean_b = np.rad2deg(2 * np.arcsin(np.sin(np.deg2rad(mean_a) / 2) * 0.9036))
    amp_b = amp_a * amp_b_ratio
    return voigt(x, amp_a, mean_a, sigma, gamma) + voigt(x, amp_b, mean_b, sigma, gamma)


def find_all_peaks_fitting(
    df: pd.DataFrame,
    initial_peaks: np.ndarray,
    bg_params: tuple,
    window: int = 20,
) -> list:
    """For each initial peak, fit a double Voigt profile on the background-subtracted data."""
    intensities = df["Intensity"].values.astype(float)
    angles = df["Angle"].values.astype(float)
    max_intensity = intensities.max() if intensities.size else 0.0

    # Subtract the global background
    background = bremsstrahlung_bg(angles, *bg_params)
    y_subtracted = intensities - background

    results = []
    for idx in initial_peaks:
        left = max(0, idx - window)
        right = min(len(df), idx + window + 1)
        x_window = angles[left:right]
        y_window = y_subtracted[left:right]
        if len(x_window) < 5:  # Need at least 5 points for 5 parameters
            results.append((idx, None, angles[idx]))
            continue
        try:
            # --- Guesses ---
            mean_guess = angles[idx]
            amplitude_guess = y_window[x_window == angles[idx]][0]
            span = max(x_window.max() - x_window.min(), 1e-6)
            sigma_guess = span / 10
            gamma_guess = sigma_guess / 2
            amp_beta_ratio_guess = 0.2
            guess = [
                amplitude_guess,
                mean_guess,
                sigma_guess,
                gamma_guess,
                amp_beta_ratio_guess,
            ]

            # --- Bounds ---
            min_sigma = max(span / 200, 1e-6)
            lower = [0.0, x_window.min(), min_sigma, min_sigma, 0.0]
            upper = [max_intensity * 2.0, x_window.max(), span, span, 1.0]

            popt, _ = curve_fit(
                double_voigt, x_window, y_window, p0=guess, bounds=(lower, upper), maxfev=20000
            )
            fit_peak = popt[1]  # mean_a parameter
            results.append((idx, popt, fit_peak))
        except Exception:
            results.append((idx, None, angles[idx]))
    return results
