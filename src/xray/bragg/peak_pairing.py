"""Peak pairing utilities for K-alpha and K-beta identification."""

import pandas as pd


def identify_ka_kb_peaks(
    peaks_data: pd.DataFrame, valid_fits: list
) -> tuple[list[float], list[float], list[tuple[float, str, int]]]:
    """
    Identify K-alpha and K-beta peaks from analysis results.

    Args:
        peaks_data: DataFrame with peak information (amplitude, angle, etc.)
        valid_fits: List of valid fit results from peak analysis

    Returns:
        Tuple of (ka_angles, kb_angles, combined_data)
        where combined_data is list of (angle, 'ka' or 'kb', order_n) tuples
    """
    # Check if amplitudes are all NaN, indicating predefined angles were used
    if peaks_data["amplitude"].isnull().all():
        return _identify_predefined_peaks(valid_fits)
    else:
        return _identify_fitted_peaks(peaks_data)


def _identify_predefined_peaks(
    valid_fits: list,
) -> tuple[list[float], list[float], list[tuple[float, str, int]]]:
    """
    Identify peaks when using predefined angles (no fitting).
    Assumes K-beta, K-alpha pairs in order.
    """
    ka_angles = []
    kb_angles = []
    combined_data = []

    order_n = 0
    for i, (_, _, mean_angle) in enumerate(valid_fits):
        if i % 2 == 0:  # Even indices (0, 2, 4, ...) are K-beta (first in pair)
            order_n += 1  # New diffraction order
            kb_angles.append(mean_angle)
            combined_data.append((mean_angle, "kb", order_n))
        else:  # Odd indices (1, 3, 5, ...) are K-alpha (second in pair)
            ka_angles.append(mean_angle)
            combined_data.append((mean_angle, "ka", order_n))  # Same order as previous K-beta

    return ka_angles, kb_angles, combined_data


def _identify_fitted_peaks(
    peaks_data: pd.DataFrame,
) -> tuple[list[float], list[float], list[tuple[float, str, int]]]:
    """
    Identify peaks from fitted data based on amplitude.
    K-alpha has higher amplitude than K-beta.
    """
    ka_angles = []
    kb_angles = []
    combined_data = []

    order_n = 0
    for i in range(0, len(peaks_data), 2):
        order_n += 1
        if i + 1 < len(peaks_data):
            # Process pair - both peaks are same diffraction order
            pair = peaks_data.iloc[i : i + 2]
            ka_peak = pair.loc[pair["amplitude"].idxmax()]
            kb_peak = pair.loc[pair["amplitude"].idxmin()]

            ka_angles.append(ka_peak["angle"])
            kb_angles.append(kb_peak["angle"])

            combined_data.append((ka_peak["angle"], "ka", order_n))
            combined_data.append((kb_peak["angle"], "kb", order_n))
        else:
            # Handle lone peak (assume it's K-alpha)
            lone_peak = peaks_data.iloc[i]
            ka_angles.append(lone_peak["angle"])
            combined_data.append((lone_peak["angle"], "ka", order_n))

    return ka_angles, kb_angles, combined_data
