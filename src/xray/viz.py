from __future__ import annotations

from pathlib import Path

import pandas as pd

# Use a soft import guard to avoid import errors in environments without matplotlib at build time
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    from scipy.special import wofz
except ImportError:
    plt = None
    np = None
    go = None
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


def create_interactive_report(
    df: pd.DataFrame,
    initial_peaks: np.ndarray,
    all_fits: list,
    bg_params: tuple | None,
    final_model_peaks: np.ndarray,
    peak_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Creates a self-contained HTML report with an interactive plot and summary tables."""
    if go is None or df.empty or np is None:
        return out_path

    fig = go.Figure()
    x_data = df["Angle"].values
    y_data = df["Intensity"].values

    # 1. Raw data
    fig.add_trace(
        go.Scatter(
            x=x_data, y=y_data, mode="markers", name="Raw Data", marker=dict(color="gray", size=4)
        )
    )

    # 2. Initial peaks
    if initial_peaks.size > 0:
        fig.add_trace(
            go.Scatter(
                x=x_data[initial_peaks],
                y=y_data[initial_peaks],
                mode="markers",
                name="Initial Peaks",
                marker=dict(color="red", size=10, symbol="x"),
            )
        )

    # 3. Global background and total fit
    if bg_params is not None:
        x_fit_global = np.linspace(x_data.min(), x_data.max(), 1000)
        y_bg_global = bremsstrahlung_bg(x_fit_global, *bg_params)
        fig.add_trace(
            go.Scatter(
                x=x_fit_global,
                y=y_bg_global,
                mode="lines",
                name="Global BG Fit",
                line=dict(color="green", dash="dash"),
            )
        )

        y_total_fit = bremsstrahlung_bg(x_data, *bg_params)
        for _, fit_params, _ in all_fits:
            if fit_params is not None:
                y_total_fit += double_voigt(x_data, *fit_params)

        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_total_fit,
                mode="lines",
                name="Total Combined Fit",
                line=dict(color="orange", width=3),
            )
        )

        # 4. Final model peaks
        if final_model_peaks.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=x_data[final_model_peaks],
                    y=y_total_fit[final_model_peaks],
                    mode="markers",
                    name="Final Model Peaks",
                    marker=dict(color="purple", size=12, symbol="cross"),
                )
            )

    fig.update_layout(
        title="X-Ray Diffraction Analysis Summary",
        xaxis_title="Angle (2θ)",
        yaxis_title="Intensity (counts)",
        legend_title="Legend",
    )

    # Convert tables to HTML
    peak_table_html = peak_table.to_html(
        classes="table table-striped table-hover", justify="center"
    )
    summary_table_html = summary_table.to_html(
        classes="table table-striped table-hover", justify="center"
    )

    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>X-Ray Analysis Report</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {{ font-family: sans-serif; padding: 2rem; }}
            .table-container {{ margin-top: 2rem; }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <h1>X-Ray Diffraction Analysis Report</h1>
            <div id="plot">{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
            <div class="table-container">
                <h2>Fitted Peak Details</h2>
                {peak_table_html}
            </div>
            <div class="table-container">
                <h2>d-spacing Summary</h2>
                {summary_table_html}
            </div>
        </div>
    </body>
    </html>
    """

    with open(out_path, "w") as f:
        f.write(html_content)

    return out_path
