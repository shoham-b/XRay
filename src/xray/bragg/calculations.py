"""Calculation utilities for Bragg diffraction analysis."""

import numpy as np
from scipy.optimize import curve_fit


def linear_fit_through_origin(x, a):
    """Linear function through origin: y = a*x"""
    return a * x


def perform_linear_fit(sin_theta_values: np.ndarray) -> float:
    """
    Perform linear fit through origin for Bragg's law analysis.

    Args:
        sin_theta_values: Array of sin(theta) values

    Returns:
        Fitted slope value
    """
    if len(sin_theta_values) == 0:
        return np.nan

    n_values = np.arange(1, len(sin_theta_values) + 1)
    popt, _ = curve_fit(linear_fit_through_origin, sin_theta_values, n_values)
    return popt[0]


def calculate_d_spacing(slope: float, wavelength: float) -> float:
    """Calculate d-spacing from fitted slope and wavelength."""
    if np.isnan(slope):
        return np.nan
    return slope * wavelength / 2


def calculate_cubic_lattice_constants(d_spacing: float) -> dict[str, float]:
    """
    Calculate lattice constants for cubic crystal systems.

    Args:
        d_spacing: The d-spacing value

    Returns:
        Dictionary with 'sc', 'bcc', 'fcc' lattice constants
    """
    if np.isnan(d_spacing):
        return {"sc": np.nan, "bcc": np.nan, "fcc": np.nan}

    return {
        "sc": d_spacing * np.sqrt(1),
        "bcc": d_spacing * np.sqrt(2),
        "fcc": d_spacing * np.sqrt(3),
    }


def calculate_error_percentage(inferred_value: float, real_value: float) -> float:
    """
    Calculate percentage error between inferred and real values.

    Args:
        inferred_value: The calculated/inferred value
        real_value: The known/real value

    Returns:
        Percentage error
    """
    if np.isnan(inferred_value) or real_value == 0:
        return np.nan
    return ((inferred_value - real_value) / real_value) * 100


def perform_bragg_fit_core(
    data_with_wavelength: list[tuple[float, float, int]],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Core Bragg fit function: n = (2d/λ)*sin(θ)

    Args:
        data_with_wavelength: List of (angle, wavelength, order_n) tuples

    Returns:
        Tuple of (sin_theta_over_lambda, n_values, slope, d_spacing)
    """
    if not data_with_wavelength:
        return np.array([]), np.array([]), np.nan, np.nan

    # Sort by angle
    sorted_data = sorted(data_with_wavelength, key=lambda x: x[0])

    # Calculate sin(theta)/lambda for each point
    # We fit: n = (2d) * (sin(θ)/λ)
    sin_theta_over_lambda_values = []
    n_values_list = []
    for angle, wavelength, order_n in sorted_data:
        sin_theta = np.sin(np.deg2rad(angle / 2))
        sin_theta_over_lambda_values.append(sin_theta / wavelength)
        n_values_list.append(order_n)

    sin_theta_over_lambda_array = np.array(sin_theta_over_lambda_values)
    n_values = np.array(n_values_list)

    # Perform fit: n = slope * (sin(θ)/λ), where slope = 2d
    if len(sin_theta_over_lambda_array) > 0:
        popt, _ = curve_fit(linear_fit_through_origin, sin_theta_over_lambda_array, n_values)
        slope = popt[0]  # slope = 2d
        d_spacing = slope / 2
    else:
        slope = np.nan
        d_spacing = np.nan

    return sin_theta_over_lambda_array, n_values, slope, d_spacing


def perform_combined_fit(
    combined_data: list[tuple[float, str, int]], lambda_a: float, lambda_b: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Perform combined fit for K-alpha and K-beta peaks together.
    Each point uses its own wavelength: n = (2d/λ)*sin(θ)

    Args:
        combined_data: List of (angle, 'ka' or 'kb', order_n) tuples
        lambda_a: K-alpha wavelength
        lambda_b: K-beta wavelength

    Returns:
        Tuple of (sin_theta_over_lambda, n_values, slope, d_spacing)
    """
    # Convert to (angle, wavelength, order_n) format
    data_with_wavelength = [
        (angle, lambda_a if peak_type == "ka" else lambda_b, order_n)
        for angle, peak_type, order_n in combined_data
    ]
    return perform_bragg_fit_core(data_with_wavelength)
