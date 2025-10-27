import numpy as np
from scipy.stats import gaussian_kde


def bragg_d_spacing(two_theta: float, wavelength: float) -> float:
    """Calculates d-spacing from the Bragg angle."""
    theta_rad = np.deg2rad(two_theta / 2.0)
    return wavelength / (2 * np.sin(theta_rad))


def find_most_probable_d(
    d_spacings: list[float],
) -> tuple[float, float, int] | None:
    """
    Finds the most probable d-spacing from a list of values using KDE.

    Returns:
        A tuple containing:
        - The most probable d-spacing (mode of the KDE).
        - The standard deviation of the input d-spacings.
        - The number of d-spacings used.
    """
    if not d_spacings or len(d_spacings) < 2:
        return None

    d_array = np.array(d_spacings, dtype=float)

    # Prefer a clear discrete mode if present (common in small peak lists)
    unique_vals, counts = np.unique(d_array, return_counts=True)
    idx_max = int(np.argmax(counts))
    has_clear_mode = counts[idx_max] >= 2 and (
        counts.size == 1 or counts[idx_max] > np.partition(counts, -2)[-2]
    )

    if has_clear_mode:
        most_probable_d = float(unique_vals[idx_max])
    else:
        # Use KDE to find the most probable value (the peak of the distribution)
        try:
            kde = gaussian_kde(d_array)
            d_range = np.linspace(d_array.min(), d_array.max(), 500)
            kde_values = kde(d_range)
            most_probable_d = float(d_range[int(np.argmax(kde_values))])
        except Exception:
            # Fallback to mean if KDE fails for any reason
            most_probable_d = float(np.mean(d_array))

    std_d = float(np.std(d_array))
    num_peaks = int(len(d_array))

    return most_probable_d, std_d, num_peaks
