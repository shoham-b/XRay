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
    x = np.asanyarray(x)
    x_shifted = x - x_offset
    x_shifted = np.where(x_shifted < 0, 0, x_shifted)
    return bg_amp * x_shifted * np.exp(-x_shifted / (bg_scale + 1e-9))


@cached
def fit_global_background(
    df: pd.DataFrame, initial_peaks: np.ndarray, window: int = 20, **kwargs
) -> tuple | None:
    """Fits a single Bremsstrahlung background to the entire spectrum, bridging over peaks."""
    x_data = df["Angle"].values
    y_data = df["Intensity"].values.copy()  # Make a copy to modify

    # Create a "bridge" over the peaks
    sorted_peaks = np.sort(initial_peaks)
    for idx in sorted_peaks:
        left = max(0, idx - window)
        right = min(len(x_data) - 1, idx + window)

        if left >= right:
            continue

        # Linearly interpolate between the edges of the window
        x1, y1 = x_data[left], y_data[left]
        x2, y2 = x_data[right], y_data[right]

        # Create the linear bridge
        m = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
        b = y1 - m * x1

        # Replace the data in the window with the bridge
        bridge_indices = np.arange(left + 1, right)
        if bridge_indices.size > 0:
            y_data[bridge_indices] = m * x_data[bridge_indices] + b

    # Now, fit the background to the modified data (with bridges)
    x_bg, y_bg = x_data, y_data
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
    """For each initial peak, fit a Voigt profile with tight bounds."""
    intensities = df["Intensity"].values.astype(float)
    angles = df["Angle"].values.astype(float)
    max_intensity = intensities.max() if intensities.size else 0.0

    results = []

    # Sort initial peaks by their index to easily find neighbors
    sorted_initial_peaks_indices = np.sort(initial_peaks)

    # Define the model to fit for each peak
    def voigt_with_bg(x, amplitude, mean, sigma, gamma):
        return voigt(x, amplitude, mean, sigma, gamma) + bremsstrahlung_bg(x, *bg_params)

    for i, idx in enumerate(sorted_initial_peaks_indices):
        try:
            left = max(0, idx - window)
            right = min(len(df), idx + window + 1)
            x_window, y_window = angles[left:right], intensities[left:right]  # Use original intensities

            # Better initial guesses
            max_intensity_in_window_idx = np.argmax(y_window)
            mean_guess = x_window[max_intensity_in_window_idx]
            amplitude_guess = max(
                0, y_window[max_intensity_in_window_idx] - bremsstrahlung_bg(mean_guess, *bg_params)
            )

            sigma_guess, gamma_guess = 0.1, 0.1

            span = max(x_window.max() - x_window.min(), 1e-6)
            guess = [amplitude_guess, mean_guess, sigma_guess, gamma_guess]

            # Set bounds for the mean based on neighboring peaks
            if i > 0:
                prev_peak_angle = angles[sorted_initial_peaks_indices[i - 1]]
                lower_bound_mean = angles[idx] - (angles[idx] - prev_peak_angle) * 0.25
            else:
                lower_bound_mean = -np.inf

            if i < len(sorted_initial_peaks_indices) - 1:
                next_peak_angle = angles[sorted_initial_peaks_indices[i + 1]]
                upper_bound_mean = angles[idx] + (next_peak_angle - angles[idx]) * 0.25
            else:
                upper_bound_mean = np.inf

            # The bounds for the mean should also be within the fitting window
            lower_bound_mean = max(lower_bound_mean, x_window.min())
            upper_bound_mean = min(upper_bound_mean, x_window.max())

            lower = [0.0, lower_bound_mean, 1e-4, 1e-4]
            upper = [max_intensity * 2.0, upper_bound_mean, span, span]

            popt, _ = curve_fit(
                voigt_with_bg, x_window, y_window, p0=guess, bounds=(lower, upper), maxfev=20000
            )
            results.append((idx, popt, popt[1]))
        except Exception:
            results.append((idx, None, angles[idx]))

    # Re-sort results to match the original order of initial_peaks
    original_order_map = {val: i for i, val in enumerate(initial_peaks)}
    results.sort(key=lambda r: original_order_map.get(r[0], -1))

    return results


def fit_spectrum_with_predefined_peaks(
    df: pd.DataFrame, predefined_peaks: list[float], **kwargs
) -> tuple | None:
    """
    Fits a composite model of background and Voigt peaks to the spectrum.

    Args:
        df: DataFrame with 'Angle' and 'Intensity' columns.
        predefined_peaks: A list of angles for the Voigt peaks.

    Returns:
        A tuple of (optimal_parameters, covariance_matrix).
    """
    x_data = df["Angle"].values
    y_data = df["Intensity"].values
    num_peaks = len(predefined_peaks)

    # 1. Define the composite model function dynamically
    def composite_model(x, *params):
        bg_amp, x_offset, bg_scale = params[0:3]
        model = bremsstrahlung_bg(x, bg_amp, x_offset, bg_scale)

        voigt_params = params[3:]
        for i in range(num_peaks):
            amp, mean, sigma, gamma = voigt_params[i * 4 : (i + 1) * 4]
            model += voigt(x, amp, mean, sigma, gamma)
        return model

    # 2. Generate initial parameter guesses
    # Background guesses
    bg_amp_guess = np.median(y_data)
    x_offset_guess = x_data.min() - (x_data.max() - x_data.min()) * 0.1
    bg_scale_guess = (x_data.max() - x_data.min()) * 2
    initial_guesses = [bg_amp_guess, x_offset_guess, bg_scale_guess]

    # Bounds for background
    lower_bounds = [0.0, -np.inf, 1e-6]
    upper_bounds = [y_data.max() * 2, x_data.max(), np.inf]

    # Voigt guesses and bounds
    for peak_angle in predefined_peaks:
        closest_idx = np.argmin(np.abs(x_data - peak_angle))
        amplitude_guess = y_data[closest_idx]
        mean_guess = peak_angle

        # Simple guess for sigma and gamma, can be improved
        sigma_guess = 0.1
        gamma_guess = 0.1

        initial_guesses.extend([amplitude_guess, mean_guess, sigma_guess, gamma_guess])

        # Bounds for Voigt parameters
        angle_span = x_data.max() - x_data.min()
        lower_bounds.extend([0.0, mean_guess - angle_span / 10, 1e-4, 1e-4])
        upper_bounds.extend(
            [y_data.max() * 2, mean_guess + angle_span / 10, angle_span / 5, angle_span / 5]
        )

    # 3. Fit the model
    try:
        popt, pcov = curve_fit(
            composite_model,
            x_data,
            y_data,
            p0=initial_guesses,
            bounds=(lower_bounds, upper_bounds),
            maxfev=100000,
        )
        return popt, pcov
    except Exception:
        return None, None


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