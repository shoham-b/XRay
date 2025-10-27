from __future__ import annotations

from pathlib import Path

import pandas as pd

# Use a soft import guard to avoid import errors in environments without matplotlib at build time
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.special import wofz
except ImportError:
    plt = None
    np = None
    wofz = None


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def voigt(x, amplitude, mean, sigma, gamma):
    """Voigt profile."""
    if np is None or wofz is None:
        return np.zeros_like(x)
    z = (x - mean + 1j * gamma) / (sigma * np.sqrt(2.0))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def bremsstrahlung_bg(x, bg_amp, x_offset, bg_scale):
    """A Maxwell-Boltzmann-like model for the Bremsstrahlung background."""
    if np is None:
        return np.zeros_like(x)
    x_shifted = x - x_offset
    x_shifted[x_shifted < 0] = 0
    return bg_amp * x_shifted * np.exp(-x_shifted / (bg_scale + 1e-9))


def double_voigt(x, amp_a, mean_a, sigma, gamma, amp_b_ratio):
    """Composite model of two Voigt profiles for K-alpha and K-beta."""
    if np is None:
        return np.zeros_like(x)
    mean_b = np.rad2deg(2 * np.arcsin(np.sin(np.deg2rad(mean_a) / 2) * 0.9036))
    amp_b = amp_a * amp_b_ratio
    return voigt(x, amp_a, mean_a, sigma, gamma) + voigt(x, amp_b, mean_b, sigma, gamma)


def plot_analysis_summary(
    df: pd.DataFrame,
    initial_peaks: np.ndarray,
    all_fits: list,
    bg_params: tuple | None,
    out_path: Path,
) -> Path:
    """Plots a comprehensive summary of the peak analysis with a global background."""
    _ensure_out_dir(out_path.parent)
    if plt is None or df.empty or np is None:
        return out_path

    fig, ax = plt.subplots(figsize=(15, 8))
    x_data = df["Angle"].values
    y_data = df["Intensity"].values

    # 1. Plot raw data
    ax.plot(x_data, y_data, ".", label="Raw Data", color="gray", alpha=0.6)

    # 2. Plot initial peaks
    if initial_peaks.size > 0:
        ax.plot(
            x_data[initial_peaks],
            y_data[initial_peaks],
            "x",
            color="red",
            markersize=10,
            mew=2,
            label="Initial Peaks",
        )

    # 3. Plot global background and total fit
    if bg_params is not None:
        x_fit_global = np.linspace(x_data.min(), x_data.max(), 1000)
        y_bg_global = bremsstrahlung_bg(x_fit_global, *bg_params)
        ax.plot(x_fit_global, y_bg_global, "--", color="green", label="Global Bremsstrahlung BG")

        y_total_fit = bremsstrahlung_bg(x_data, *bg_params)
        fit_plotted = False
        for _, fit_params, _ in all_fits:
            if fit_params is not None:
                y_total_fit += double_voigt(x_data, *fit_params)

                # Add markers for the fitted peak positions
                amp_a, mean_a, _, _, amp_b_ratio = fit_params
                mean_b = np.rad2deg(2 * np.arcsin(np.sin(np.deg2rad(mean_a) / 2) * 0.9036))
                ax.axvline(
                    x=mean_a,
                    color="blue",
                    linestyle="--",
                    linewidth=1,
                    label="Fitted K-a" if not fit_plotted else None,
                )
                ax.axvline(
                    x=mean_b,
                    color="purple",
                    linestyle=":",
                    linewidth=1,
                    label="Fitted K-β" if not fit_plotted else None,
                )
                fit_plotted = True

        ax.plot(x_data, y_total_fit, "-", color="orange", linewidth=2.5, label="Total Combined Fit")

    ax.set_xlabel("Angle (2θ)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title("X-Ray Diffraction Analysis Summary")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlim(x_data.min(), x_data.max())

    fig.tight_layout()
    fig.savefig(out_path.as_posix(), dpi=150)
    plt.close(fig)
    return out_path
