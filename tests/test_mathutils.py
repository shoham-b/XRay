import numpy as np
import pytest

from xray.mathutils import bragg_d_spacing, find_most_probable_d


def test_bragg_d_spacing():
    # Test with known values
    assert bragg_d_spacing(two_theta=20, wavelength=1.54) == pytest.approx(4.435, abs=1e-3)

    # Test with two_theta = 180
    assert bragg_d_spacing(two_theta=180, wavelength=1.54) == pytest.approx(0.77)

    # Test with two_theta = 0, which should result in infinity
    assert bragg_d_spacing(two_theta=0, wavelength=1.54) == np.inf


def test_find_most_probable_d_empty_list():
    assert find_most_probable_d([]) is None


def test_find_most_probable_d_single_value():
    assert find_most_probable_d([1.0]) is None


def test_find_most_probable_d_mode_shortcut():
    # Tests the mode shortcut for clean data
    d_spacings = [1.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    most_probable_d, std_d, num_peaks = find_most_probable_d(d_spacings)
    assert most_probable_d == pytest.approx(3.0, abs=1e-1)
    assert std_d == pytest.approx(np.std(d_spacings))
    assert num_peaks == len(d_spacings)


def test_find_most_probable_d_identical_values():
    # Tests the mode shortcut with identical values
    d_spacings = [2.0, 2.0, 2.0, 2.0]
    most_probable_d, std_d, num_peaks = find_most_probable_d(d_spacings)
    assert most_probable_d == pytest.approx(2.0)
    assert std_d == pytest.approx(0.0)
    assert num_peaks == 4


def test_find_most_probable_d_with_curve_fit():
    # Tests a successful curve_fit case
    d_spacings = [1.4, 1.4, 1.5, 1.6, 1.6]
    most_probable_d, std_d, num_peaks = find_most_probable_d(d_spacings)
    assert most_probable_d == pytest.approx(1.5, abs=0.2)
    assert std_d == pytest.approx(np.std(d_spacings))
    assert num_peaks == len(d_spacings)
