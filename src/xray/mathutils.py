import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import mode


def bragg_d_spacing(two_theta: float, wavelength: float) -> float:
    """Calculates d-spacing from the Bragg angle."""
    theta_rad = np.deg2rad(two_theta / 2.0)
    return wavelength / (2 * np.sin(theta_rad))


def gaussian(x: np.ndarray, amplitude: float, mean: float, stddev: float) -> np.ndarray:
    """Gaussian function."""
    return amplitude * np.exp(-(((x - mean) / stddev) ** 2) / 2)


def find_most_probable_d(
    d_spacings: list[float],
) -> tuple[float, float, int] | None:
    """
    Finds the most probable d-spacing by fitting a Gaussian to the histogram of values.

    Returns:
        A tuple containing:
        - The most probable d-spacing (mean of the fitted Gaussian).
        - The standard deviation of the input d-spacings.
        - The number of d-spacings used.
    """
    if not d_spacings or len(d_spacings) < 2:
        return None

    d_array = np.array(d_spacings)
    num_peaks = len(d_array)
    std_d = float(np.std(d_array))

    # If there's a very clear mode (at least half the peaks are the same), just use that.
    # This is robust for very clean data with few peaks.
    mode_res = mode(d_array)
    if mode_res.count and mode_res.count[0] >= num_peaks / 2 and num_peaks > 2:
        return float(mode_res.mode[0]), std_d, num_peaks

    try:
        # Create a histogram of the d-spacing values
        num_bins = min(max(10, num_peaks // 2), 50)
        counts, bin_edges = np.histogram(d_array, bins=num_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Initial parameter guesses for the Gaussian fit
        p0 = [np.max(counts), np.mean(d_array), std_d]

        # Fit the Gaussian function to the histogram data
        params, _ = curve_fit(gaussian, bin_centers, counts, p0=p0)
        most_probable_d = float(params[1])  # The mean of the fitted Gaussian

    except (RuntimeError, ValueError):
        # If curve_fit fails (e.g., not enough data, poor initial guess),
        # fall back to a simple mean.
        most_probable_d = float(np.mean(d_array))

    return most_probable_d, std_d, num_peaks
