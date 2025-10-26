"""Unified math utilities for X-Ray analysis.

This module centralizes small math helpers used across the project to avoid
duplicating functionality in subpackages.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def normalized_sum(values: Iterable[float]) -> float:
    """Return the sum of values after normalizing them into [0, 1] range.

    If the iterable is empty, returns 0.0.
    If all values are equal, returns 0.0 (since normalization would divide by zero otherwise).
    """
    vals = list(values)
    if not vals:
        return 0.0
    vmin = min(vals)
    vmax = max(vals)
    span = vmax - vmin
    if span == 0:
        return 0.0

    # Handle extreme floating-point values that could cause NaN
    if not math.isfinite(span) or span == 0:
        return 0.0

    total = 0.0
    for v in vals:
        if math.isfinite(v):
            normalized = (v - vmin) / span
            if math.isfinite(normalized):
                total += normalized
    return total


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp a value into the [lo, hi] range."""
    return max(lo, min(hi, v))


def safe_log(x: float, eps: float = 1e-12) -> float:
    """Numerically safe natural log guarding against non-positive inputs."""
    return math.log(max(x, eps))


def gaussian(x, amplitude, mean, stddev):
    """
    Gaussian function.
    """
    return amplitude * np.exp(-(((x - mean) / stddev) ** 2) / 2)


def bragg_d_spacing(two_theta_degrees: float, wavelength_angstroms: float) -> float:
    """
    Calculates d-spacing using Bragg's Law for first-order diffraction (n=1).

    Args:
        two_theta_degrees: The diffraction angle (2-theta) in degrees.
        wavelength_angstroms: The wavelength of the X-ray source in Angstroms.

    Returns:
        The d-spacing in Angstroms.
    """
    theta_radians = math.radians(two_theta_degrees / 2)
    if theta_radians == 0:
        return float("inf")
    return wavelength_angstroms / (2 * math.sin(theta_radians))
