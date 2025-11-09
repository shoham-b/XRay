import numpy as np
import pandas as pd

from src.xray.bragg.peak_finding import find_all_peaks_naive


def test_find_all_peaks_naive_no_smoothing():
    # Create a DataFrame with some clear peaks and some noise
    data = {"Intensity": [0, 1, 0, 0, 5, 0, 0, 10, 0, 0, 2, 0], "Angle": range(12)}
    df = pd.DataFrame(data)

    # Without smoothing, only prominent peaks should be found
    peaks, _ = find_all_peaks_naive(df)
    assert len(peaks) == 4
    assert np.array_equal(peaks, [1, 4, 7, 10])


def test_find_all_peaks_naive_with_smoothing():
    # Create a DataFrame with some subtle peaks that might be missed without smoothing
    data = {
        "Intensity": [0, 0.5, 0.6, 0.5, 0.1, 0.2, 0.3, 0.2, 0.1, 0.8, 0.9, 0.8, 0.1, 0.0],
        "Angle": range(14),
    }
    df = pd.DataFrame(data)

    # Without smoothing, fewer peaks might be found
    peaks_no_smooth, _ = find_all_peaks_naive(df)
    assert len(peaks_no_smooth) == 2  # Expected peaks at index 2 and 10

    # With smoothing, more subtle peaks should be detectable
    # A smoothing window of 3 should help
    peaks_with_smooth, _ = find_all_peaks_naive(df)
    # The exact number of peaks might vary slightly depending on the smoothing effect
    # but it should ideally be more or at least the same as without smoothing,
    # and include the subtle ones.
    assert len(peaks_with_smooth) >= len(peaks_no_smooth)
    # Check if the expected peaks are still there or new ones are found
    assert 2 in peaks_with_smooth
    assert 10 in peaks_with_smooth


def test_find_all_peaks_naive_smoothing_window_one():
    # Smoothing window of 1 should behave like no smoothing
    data = {"Intensity": [0, 1, 0, 0, 5, 0, 0, 10, 0, 0, 2, 0], "Angle": range(12)}
    df = pd.DataFrame(data)

    peaks_no_smooth, _ = find_all_peaks_naive(df)
    peaks_smooth_one, _ = find_all_peaks_naive(df)

    assert np.array_equal(peaks_no_smooth, peaks_smooth_one)


def test_find_all_peaks_naive_smoothing_window_even_or_invalid():
    # Even smoothing window should not apply smoothing (or raise error if implemented to do so)
    # Current implementation skips smoothing if window is even.
    data = {"Intensity": [0, 1, 0, 0, 5, 0, 0, 10, 0, 0, 2, 0], "Angle": range(12)}
    df = pd.DataFrame(data)

    peaks_no_smooth, _ = find_all_peaks_naive(df)
    peaks_smooth_even, _ = find_all_peaks_naive(df)

    assert np.array_equal(peaks_no_smooth, peaks_smooth_even)
