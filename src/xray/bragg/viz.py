from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import wofz


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


def _create_single_material_plot(
    df: pd.DataFrame,
    analysis_results: dict,
    peak_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    material_name: str,
    fit_plot_data: dict,
) -> go.Figure:
    """Creates an interactive plot for a single material."""

    rows = 4
    row_heights = [0.5, 0.16, 0.16, 0.18]
    subplot_titles = [
        f"X-Ray Diffraction Spectrum - {material_name}",
        "d-spacing Linear Fit (n vs sinθ) - Kα",
        "d-spacing Linear Fit (n vs sinθ) - Kβ",
        "d-spacing Linear Fit (n vs sinθ) - Combined (Kα + Normalized Kβ)",
    ]

    fig = make_subplots(
        rows=rows,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        row_heights=row_heights,
    )

    x_data = df["Angle"].values
    y_data = df["Intensity"].values
    y_error = np.sqrt(y_data)

    # 1. Raw data
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            name="Raw Data",
            marker=dict(color="gray", size=4, opacity=0.6),
        ),
        row=1,
        col=1,
    )

    # 2. Initial peaks (these are just markers, no error bars)
    initial_peaks = analysis_results.get("initial_peaks_idx", np.array([]))
    if initial_peaks.size > 0:
        fig.add_trace(
            go.Scatter(
                x=x_data[initial_peaks],
                y=y_data[initial_peaks],
                mode="markers",
                name="Initial Peaks",
                marker=dict(color="red", size=10, symbol="x", line=dict(width=1, color="black")),
            ),
            row=1,
            col=1,
        )

    # Get background parameters first, as they are needed for peak y-values
    bg_params = analysis_results.get("bg_params")

    # 3. Fitted Peaks with error bars (new trace)
    if not peak_table.empty and bg_params is not None:
        # Calculate y-position of fitted peaks on top of the background
        peak_y = peak_table["Amplitude"] + bremsstrahlung_bg(peak_table["Angle"].values, *bg_params)
        fig.add_trace(
            go.Scatter(
                x=peak_table["Angle"],
                y=peak_y,
                mode="markers",
                name="Fitted Peaks",
                error_x=dict(
                    type="data",
                    array=peak_table["Sigma"],
                    visible=True,
                    thickness=1,
                    width=2,
                    color="blue",
                ),
                marker=dict(
                    color="blue", size=10, symbol="circle", line=dict(width=1, color="black")
                ),
            ),
            row=1,
            col=1,
        )

    # 4. Global background and total fit
    if bg_params is not None:
        x_fit_global = np.linspace(x_data.min(), x_data.max(), 1000)
        y_bg_global = bremsstrahlung_bg(x_fit_global, *bg_params)

        fig.add_trace(
            go.Scatter(
                x=x_fit_global,
                y=y_bg_global,
                mode="lines",
                name="Background Fit",
                line=dict(color="green", dash="dash", width=2),
            ),
            row=1,
            col=1,
        )

        y_total_fit = bremsstrahlung_bg(x_data, *bg_params)
        all_fits = analysis_results.get("valid_fits", [])
        for i, (_, fit_params, _) in enumerate(all_fits):
            if fit_params is not None:
                y_peak = voigt(x_data, *fit_params)
                y_total_fit += y_peak
                # Add individual peak trace
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_peak,
                        mode="lines",
                        name=f"Voigt Peak {i+1}",
                        line=dict(width=1.5, dash="dot"),
                        showlegend=True,
                        visible="legendonly",
                    ),
                    row=1,
                    col=1,
                )

        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_total_fit,
                mode="lines",
                name="Total Fit",
                line=dict(color="red", width=2.5),
            ),
            row=1,
            col=1,
        )

    # 5. d-spacing linear fit plots
    if fit_plot_data:
        # K-alpha fit
        ka_x_values = fit_plot_data["ka_x_values"]
        ka_y_values = fit_plot_data["ka_y_values"]
        ka_slope = fit_plot_data["ka_slope"]

        fig.add_trace(
            go.Scatter(
                x=ka_x_values,
                y=ka_y_values,
                mode="markers",
                name="Kα Data",
                marker=dict(color="blue"),
            ),
            row=2,
            col=1,
        )
        ka_fit_line_y = ka_slope * ka_x_values
        fig.add_trace(
            go.Scatter(
                x=ka_x_values,
                y=ka_fit_line_y,
                mode="lines",
                name=f"Kα Fit (slope={ka_slope:.4f})",
                line=dict(color="blue", dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(
            title_text="sin(θ)",
            range=[ka_x_values.min() - 0.01, ka_x_values.max() + 0.01],
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Peak Order (n)", row=2, col=1)

        # K-beta fit
        kb_x_values = fit_plot_data["kb_x_values"]
        kb_y_values = fit_plot_data["kb_y_values"]
        kb_slope = fit_plot_data["kb_slope"]

        fig.add_trace(
            go.Scatter(
                x=kb_x_values,
                y=kb_y_values,
                mode="markers",
                name="Kβ Data",
                marker=dict(color="green"),
            ),
            row=3,
            col=1,
        )
        kb_fit_line_y = kb_slope * kb_x_values
        fig.add_trace(
            go.Scatter(
                x=kb_x_values,
                y=kb_fit_line_y,
                mode="lines",
                name=f"Kβ Fit (slope={kb_slope:.4f})",
                line=dict(color="green", dash="dash"),
            ),
            row=3,
            col=1,
        )
        fig.update_xaxes(
            title_text="sin(θ)",
            range=[kb_x_values.min() - 0.01, kb_x_values.max() + 0.01],
            row=3,
            col=1,
        )
        fig.update_yaxes(title_text="Peak Order (n)", row=3, col=1)

        # Combined fit
        combined_x_values = fit_plot_data["combined_x_values"]
        combined_y_values = fit_plot_data["combined_y_values"]
        combined_slope = fit_plot_data["combined_slope"]

        fig.add_trace(
            go.Scatter(
                x=combined_x_values,
                y=combined_y_values,
                mode="markers",
                name="Combined Data",
                marker=dict(color="purple"),
            ),
            row=4,
            col=1,
        )
        combined_fit_line_y = combined_slope * combined_x_values
        fig.add_trace(
            go.Scatter(
                x=combined_x_values,
                y=combined_fit_line_y,
                mode="lines",
                name=f"Combined Fit (slope={combined_slope:.4f})",
                line=dict(color="purple", dash="dash"),
            ),
            row=4,
            col=1,
        )
        all_x_values_combined = np.concatenate((ka_x_values, kb_x_values))
        fig.update_xaxes(
            title_text="sin(θ)",
            range=[all_x_values_combined.min() - 0.01, all_x_values_combined.max() + 0.01],
            row=4,
            col=1,
        )
        fig.update_yaxes(title_text="Peak Order (n)", row=4, col=1)

    fig.update_layout(
        height=1000,
        showlegend=True,
    )
    fig.update_xaxes(title_text="2θ (degrees)", range=[x_data.min(), x_data.max()], row=1, col=1)
    fig.update_yaxes(title_text="Intensity (photons/second)", row=1, col=1)

    return fig


def create_multi_material_report(
    analysis_data_list: list[dict],
    out_path: Path,
) -> Path:
    """Creates a self-contained HTML report with tabs for multiple materials."""
    if not analysis_data_list:
        return out_path

    tab_headers = []
    tab_contents = []

    # Use a single include_plotlyjs=True for the first plot, then False for the rest.
    include_plotlyjs = True

    for i, analysis_data in enumerate(analysis_data_list):
        material_name = analysis_data["name"]
        df = analysis_data["df"]
        analysis_results = analysis_data["analysis_results"]
        peak_table = analysis_data["peak_df"]
        summary_table = analysis_data["summary_df"]
        fit_plot_data = analysis_data["fit_plot_data"]

        fig = _create_single_material_plot(
            df, analysis_results, peak_table, summary_table, material_name, fit_plot_data
        )

        plot_html = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
        if include_plotlyjs:
            include_plotlyjs = False

        peak_table_html = peak_table.to_html(
            classes="table table-striped table-hover", justify="center"
        )
        summary_table_html = summary_table.to_html(
            classes="table table-striped table-hover", justify="center"
        )

        active_class = "active show" if i == 0 else ""
        fade_class = "" if i == 0 else "fade"
        tab_headers.append(
            f'<li class="nav-item"><a class="nav-link {active_class}" data-toggle="tab" href="#{material_name}">{material_name}</a></li>'
        )
        tab_contents.append(f"""
            <div class="tab-pane container-fluid {fade_class} {active_class}" id="{material_name}">
                <div id="plot_{material_name}">{plot_html}</div>
            </div>
        """)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>X-Ray Analysis Report</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        <style>
            body {{ font-family: sans-serif; padding: 2rem; }}
            .table-container {{ margin-top: 2rem; }}
            .nav-tabs {{ margin-bottom: 1rem; }}
            .plotly-graph-div {{ width: 100% !important; }}
            .js-plotly-plot {{ width: 100% !important; }}
            .plot-container {{ width: 100%; }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <h1>X-Ray Diffraction Analysis Report</h1>
            <ul class="nav nav-tabs">
                {''.join(tab_headers)}
            </ul>
            <div class="tab-content">
                {''.join(tab_contents)}
            </div>
        </div>
        <script>
            // Function to handle plot resizing
            function resizePlot(tabPane) {{
                var plotDiv = $(tabPane).find(".plotly-graph-div")[0];
                if (plotDiv && typeof Plotly !== 'undefined') {{
                    // Force the plot to take full width of its container
                    Plotly.relayout(plotDiv, {{
                        'autosize': true,
                        'width': null,
                        'margin': {{l: 60, r: 20, t: 20, b: 60}},
                        'height': 1000
                    }});
                    
                    // Force a redraw to ensure proper rendering
                    setTimeout(function() {{
                        Plotly.Plots.resize(plotDiv);
                    }}, 100);
                }}
            }}

            // Handle tab changes
            $('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {{
                var target = $(e.target).attr("href"); // activated tab
                resizePlot(target);
                // Force a resize after a short delay to ensure the tab is fully shown
                setTimeout(function() {{
                    resizePlot(target);
                }}, 100);
            }});

            // Resize all plots on window resize with debounce
            var resizeTimer;
            $(window).on('resize', function() {{
                clearTimeout(resizeTimer);
                resizeTimer = setTimeout(function() {{
                    $('.tab-pane.active').each(function() {{
                        resizePlot(this);
                    }});
                }}, 250);
            }});

            // Initial resize for the active tab after everything is loaded
            $(window).on('load', function() {{
                var activeTab = $('.tab-pane.active')[0];
                if (activeTab) {{
                    // Multiple resizes to ensure proper rendering
                    resizePlot(activeTab);
                    setTimeout(function() {{ resizePlot(activeTab); }}, 100);
                    setTimeout(function() {{ resizePlot(activeTab); }}, 300);
                }}
            }});
        </script>
    </body>
    </html>
    """

    _ensure_out_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return out_path
