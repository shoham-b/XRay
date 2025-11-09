import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz

from xray.cache import cached


def find_all_peaks_naive(
    df: pd.DataFrame,
    threshold: float = 0.0,
    distance: int = 5,
    prominence: float = 0.05,
    width: int = 1,
) -> tuple[np.ndarray, dict]:
    """Finds all local maxima (spikes) in the data that meet specified criteria.

    Args:
        df: DataFrame containing 'Intensity' column with the data
        threshold: Absolute intensity threshold (default: 0.0, use prominence instead)
        distance: Minimum distance between peaks in data points
        prominence: Minimum prominence of peaks (relative to data range)
        width: Minimum width of peaks in data points

    Returns:
        Tuple of (peak_indices, peak_properties)
    """
    intensities = df["Intensity"].values
    data_range = np.ptp(intensities)  # Peak-to-peak range
    min_prominence = data_range * prominence if data_range > 0 else 0.1

    peaks, properties = find_peaks(
        intensities,
        height=threshold,
        distance=distance,
        prominence=min_prominence,
        width=width,
        rel_height=0.5,
    )

    return peaks, properties


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
    df: pd.DataFrame, initial_peaks: np.ndarray, window: int = 20, **kwargs
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


def two_voigts(x, amp1, mean1, sigma1, gamma1, amp2, mean2, sigma2, gamma2):
    """Sum of two independent Voigt profiles."""
    return voigt(x, amp1, mean1, sigma1, gamma1) + voigt(x, amp2, mean2, sigma2, gamma2)


@cached
def find_all_peaks_fitting(
    df: pd.DataFrame,
    initial_peaks: np.ndarray,
    initial_peaks_properties: dict,
    bg_params: tuple,
    window: int = 20,
) -> list:
    """For each pair of initial peaks, fit a sum of two Voigt profiles with tight bounds."""
    intensities = df["Intensity"].values.astype(float)
    angles = df["Angle"].values.astype(float)
    max_intensity = intensities.max() if intensities.size else 0.0
    background = bremsstrahlung_bg(angles, *bg_params)
    y_subtracted = intensities - background

    results = []

    peak_props_map = {peak: {} for peak in initial_peaks}
    for key, values in initial_peaks_properties.items():
        for i, peak in enumerate(initial_peaks):
            peak_props_map[peak][key] = values[i]

    sorted_indices = sorted(initial_peaks)

    for i in range(0, len(sorted_indices), 2):
        idx1 = sorted_indices[i]
        if i + 1 >= len(sorted_indices):
            # Handle lone peak with a single Voigt fit
            try:
                left = max(0, idx1 - window)
                right = min(len(df), idx1 + window + 1)
                x_window, y_window = angles[left:right], y_subtracted[left:right]

                mean_guess = angles[idx1]
                amplitude_guess = y_window[x_window == angles[idx1]][0]
                width_guess = peak_props_map[idx1].get(
                    "widths", (x_window.max() - x_window.min()) / 2
                )
                width_guess_angle = width_guess * np.mean(np.diff(angles))
                sigma_guess, gamma_guess = width_guess_angle / 5, width_guess_angle / 10

                span = max(x_window.max() - x_window.min(), 1e-6)
                guess = [amplitude_guess, mean_guess, sigma_guess, gamma_guess]
                lower = [0.0, mean_guess - span / 4, span / 200, span / 200]
                upper = [max_intensity * 2.0, mean_guess + span / 4, span / 2, span / 2]

                popt, _ = curve_fit(
                    voigt, x_window, y_window, p0=guess, bounds=(lower, upper), maxfev=20000
                )
                results.append((idx1, popt, popt[1]))
            except Exception:
                results.append((idx1, None, angles[idx1]))
            continue

        idx2 = sorted_indices[i + 1]

        left = max(0, idx1 - window)
        right = min(len(df), idx2 + window + 1)
        x_window, y_window = angles[left:right], y_subtracted[left:right]

        if len(x_window) < 8:
            results.append((idx1, None, angles[idx1]))
            results.append((idx2, None, angles[idx2]))
            continue

        try:
            mean1_guess = angles[idx1]
            amp1_guess = y_window[x_window == angles[idx1]][0]

            mean2_guess = angles[idx2]
            amp2_guess = y_window[x_window == angles[idx2]][0]

            span = max(x_window.max() - x_window.min(), 1e-6)
            sigma_guess, gamma_guess = span / 10, span / 20

            guess = [
                amp1_guess,
                mean1_guess,
                sigma_guess,
                gamma_guess,
                amp2_guess,
                mean2_guess,
                sigma_guess,
                gamma_guess,
            ]

            lower = [
                0.0,
                mean1_guess - span / 4,
                span / 200,
                span / 200,
                0.0,
                mean2_guess - span / 4,
                span / 200,
                span / 200,
            ]
            upper = [
                max_intensity * 2,
                mean1_guess + span / 4,
                span / 2,
                span / 2,
                max_intensity * 2,
                mean2_guess + span / 4,
                span / 2,
                span / 2,
            ]

            popt, _ = curve_fit(
                two_voigts, x_window, y_window, p0=guess, bounds=(lower, upper), maxfev=50000
            )

            popt1 = popt[0:4]
            popt2 = popt[4:8]

            results.append((idx1, popt1, popt1[1]))
            results.append((idx2, popt2, popt2[1]))

        except Exception:
            results.append((idx1, None, angles[idx1]))
            results.append((idx2, None, angles[idx2]))

    return results


def get_predefined_peaks(df: pd.DataFrame, predefined_angles: list[float]) -> list:
    """Returns predefined angles in a format consistent with peak-finding functions.

    Args:
        df: DataFrame containing 'Angle' column.
        predefined_angles: A list of angles to be treated as peaks.

    Returns:
        A list of tuples, where each tuple is (closest_idx, None, angle).
    """
    angles = df["Angle"].values
    results = []
    for angle in predefined_angles:
        closest_idx = np.argmin(np.abs(angles - angle))
        results.append((closest_idx, (None, angle, None, None), angle))
    return results
