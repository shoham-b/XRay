import numpy as np
import pandas as pd
import pytest

from xray.analysis.peak_finding import (
    bremsstrahlung_bg,
    double_voigt,
    find_all_peaks_fitting,
    find_all_peaks_naive,
    fit_global_background,
)


def make_df(angles, intensities):
    return pd.DataFrame({"Angle": angles, "Intensity": intensities})


@pytest.fixture
def synthetic_data():
    """Generate synthetic data with two peaks on a background."""
    angles = np.linspace(10, 80, 1000)
    # Background
    bg_params = (500, 10, 30)  # amp, offset, scale
    intensities = bremsstrahlung_bg(angles, *bg_params)
    # Peaks
    peak1_params = (2000, 30, 0.5, 0.5, 0.5)  # amp, mean, sigma, gamma, ratio
    peak2_params = (1500, 60, 0.6, 0.4, 0.4)
    intensities += double_voigt(angles, *peak1_params)
    intensities += double_voigt(angles, *peak2_params)
    return make_df(angles, intensities)


def test_find_all_peaks_naive(synthetic_data):
    peaks = find_all_peaks_naive(synthetic_data, threshold=0.2, prominence=0.1, distance=100)
    assert len(peaks) == 2
    # Check that the found peaks are close to the true means (30 and 60)
    found_angles = synthetic_data["Angle"].iloc[peaks].values
    assert np.allclose(found_angles, [30, 60], atol=1.0)


def test_fit_global_background(synthetic_data):
    initial_peaks = find_all_peaks_naive(
        synthetic_data, threshold=0.2, prominence=0.1, distance=100
    )
    bg_params = fit_global_background(synthetic_data, initial_peaks, window=50)
    assert bg_params is not None
    assert len(bg_params) == 3
    # Check that the fitted params are reasonably close to the true ones (500, 10, 30)
    assert np.allclose(bg_params, [500, 10, 30], atol=150)


def test_find_all_peaks_fitting(synthetic_data):
    initial_peaks = find_all_peaks_naive(
        synthetic_data, threshold=0.2, prominence=0.1, distance=100
    )
    bg_params = fit_global_background(synthetic_data, initial_peaks, window=50)
    assert bg_params is not None

    fits = find_all_peaks_fitting(synthetic_data, initial_peaks, bg_params, window=50)
    assert len(fits) == 2

    valid_fits = [f for f in fits if f[1] is not None]
    assert len(valid_fits) == 2

    # Check the first fitted peak
    fit1_params = valid_fits[0][1]
    assert len(fit1_params) == 5
    assert np.allclose(fit1_params[1], 30, atol=0.5)  # mean_a

    # Check the second fitted peak
    fit2_params = valid_fits[1][1]
    assert len(fit2_params) == 5
    assert np.allclose(fit2_params[1], 60, atol=0.5)  # mean_a


def test_empty_df_raises_error():
    df = make_df([], [])
    with pytest.raises(ValueError, match="cannot be empty"):
        find_all_peaks_naive(df)
