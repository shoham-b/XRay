from __future__ import annotations

from pathlib import Path

import pandas as pd

# Use a soft import guard to avoid import errors in environments without matplotlib at build time
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as _e:  # pragma: no cover - plotting backend issues are environment-specific
    plt = None  # type: ignore


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_scan(df: pd.DataFrame, out_path: Path) -> Path:
    """Plots the X-ray scan data (Intensity vs. Angle).

    Args:
        df: DataFrame with 'Angle' and 'Intensity' columns.
        out_path: Path to save the plot.

    Returns:
        Path to the saved plot.
    """
    _ensure_out_dir(out_path.parent)
    if plt is None or df.empty:
        return out_path

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Angle"], df["Intensity"], label="Intensity")
    ax.set_xlabel("Angle (2θ)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title("X-Ray Diffraction Scan")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path.as_posix(), dpi=150)
    plt.close(fig)
    return out_path


def gaussian(x, amplitude, mean, stddev):
    import numpy as np

    return amplitude * np.exp(-(((x - mean) / stddev) ** 2) / 2)


def plot_peak_fit(df: pd.DataFrame, fit_params: tuple, out_path: Path) -> Path:
    """Plots the raw data and the fitted Gaussian peak.

    Args:
        df: DataFrame with 'Angle' and 'Intensity' columns.
        fit_params: Tuple of fitted Gaussian parameters (amplitude, mean, stddev).
        out_path: Path to save the plot.

    Returns:
        Path to the saved plot.
    """
    _ensure_out_dir(out_path.parent)
    if plt is None or df.empty:
        return out_path

    try:
        import numpy as np
    except ImportError:
        return out_path  # numpy is required for this plot

    x_data = df["Angle"]
    y_data = df["Intensity"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_data, y_data, "o", label="Data")

    # Generate points for the fitted curve
    x_fit = np.linspace(x_data.min(), x_data.max(), 200)
    y_fit = gaussian(x_fit, *fit_params)
    ax.plot(x_fit, y_fit, "-", label="Fit")

    ax.set_xlabel("Angle (2θ)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title("Peak Fit")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path.as_posix(), dpi=150)
    plt.close(fig)
    return out_path
