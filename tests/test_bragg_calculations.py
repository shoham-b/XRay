import numpy as np
import pytest

from xray.bragg.calculations import calculate_error_percentage, calculate_std_dev_from_uncertainty


def test_calculate_error_percentage():
    # Test with positive values
    assert calculate_error_percentage(105, 100) == pytest.approx(5.0)
    assert calculate_error_percentage(95, 100) == pytest.approx(-5.0)
    assert calculate_error_percentage(100, 100) == pytest.approx(0.0)

    # Test with zero real_value (should return nan)
    assert np.isnan(calculate_error_percentage(100, 0))
    assert np.isnan(calculate_error_percentage(0, 0))

    # Test with nan inferred_value (should return nan)
    assert np.isnan(calculate_error_percentage(np.nan, 100))

    # Test with negative values
    assert calculate_error_percentage(-105, -100) == pytest.approx(5.0)
    assert calculate_error_percentage(-95, -100) == pytest.approx(-5.0)


def test_calculate_std_dev_from_uncertainty():
    # Test with positive deviation
    assert calculate_std_dev_from_uncertainty(105, 100, 2.5) == pytest.approx(2.0)
    # Test with negative deviation
    assert calculate_std_dev_from_uncertainty(95, 100, 2.5) == pytest.approx(-2.0)
    # Test with zero deviation
    assert calculate_std_dev_from_uncertainty(100, 100, 2.5) == pytest.approx(0.0)

    # Test with uncertainty = 0 (should return nan)
    assert np.isnan(calculate_std_dev_from_uncertainty(105, 100, 0))
    # Test with uncertainty = nan (should return nan)
    assert np.isnan(calculate_std_dev_from_uncertainty(105, 100, np.nan))
    # Test with inferred_value = nan (should return nan)
    assert np.isnan(calculate_std_dev_from_uncertainty(np.nan, 100, 2.5))

    # Test with negative values
    assert calculate_std_dev_from_uncertainty(-95, -100, 2.5) == pytest.approx(2.0)
    assert calculate_std_dev_from_uncertainty(-105, -100, 2.5) == pytest.approx(-2.0)
