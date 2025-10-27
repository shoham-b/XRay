import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz

from xray.cache import cached


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


@cached
def fit_global_background(
    df: pd.DataFrame, initial_peaks: np.ndarray, window: int = 20
) -> tuple | None:
    """Fits a single Bremsstrahlung background to the entire spectrum."""
    x_data = df["Angle"].values
    y_data = df["Intensity"].values

    mask = np.ones_like(x_data, dtype=bool)
    for idx in initial_peaks:
        left = max(0, idx - window)
        right = min(len(x_data), idx + window + 1)
        mask[left:right] = False

    x_bg, y_bg = x_data[mask], y_data[mask]
    if len(x_bg) < 3:
        return None

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


@cached
def find_all_peaks_fitting(
    df: pd.DataFrame,
    initial_peaks: np.ndarray,
    bg_params: tuple,
    window: int = 20,
) -> list:
    """For each K-a/K-b pair, fit a double Voigt profile on the background-subtracted data."""
    intensities = df["Intensity"].values.astype(float)
    angles = df["Angle"].values.astype(float)
    max_intensity = intensities.max() if intensities.size else 0.0
    background = bremsstrahlung_bg(angles, *bg_params)
    y_subtracted = intensities - background

    results = []
    sorted_indices = sorted(initial_peaks)

    # Iterate through peaks in pairs (K-alpha, K-beta)
    for i in range(0, len(sorted_indices), 2):
        ka_idx = sorted_indices[i]
        # Define a window around the pair
        if i + 1 < len(sorted_indices):
            kb_idx = sorted_indices[i + 1]
            left = max(0, ka_idx - window)
            right = min(len(df), kb_idx + window + 1)
        else:  # Handle a lone peak
            left = max(0, ka_idx - window)
            right = min(len(df), ka_idx + window + 1)

        x_window, y_window = angles[left:right], y_subtracted[left:right]
        if len(x_window) < 5:
            results.append((ka_idx, None, angles[ka_idx]))
            continue

        try:
            # Use the K-alpha peak for the initial guess
            mean_guess = angles[ka_idx]
            amplitude_guess = y_window[x_window == angles[ka_idx]][0]
            span = max(x_window.max() - x_window.min(), 1e-6)
            sigma_guess, gamma_guess = span / 10, span / 20
            guess = [amplitude_guess, mean_guess, sigma_guess, gamma_guess, 0.5]

            lower = [0.0, x_window.min(), span / 200, span / 200, 0.0]
            upper = [max_intensity * 2.0, x_window.max(), span, span, 1.0]

            popt, _ = curve_fit(
                double_voigt, x_window, y_window, p0=guess, bounds=(lower, upper), maxfev=20000
            )
            results.append((ka_idx, popt, popt[1]))
        except Exception:
            results.append((ka_idx, None, angles[ka_idx]))
    return results
