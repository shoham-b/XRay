from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from scipy.signal import find_peaks

from xray.lau.peak_finding import (
    bremsstrahlung_bg,
    double_voigt,
    find_all_peaks_fitting,
    find_all_peaks_naive,
    fit_global_background,
)
from xray.mathutils import bragg_d_spacing, find_most_probable_d


def load_and_prep_data(input_path: Path, console: Console) -> pd.DataFrame | None:
    """Loads and prepares the XRD data from a CSV file."""
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        console.print(f"[bold red]Error: Input file not found at {input_path}[/bold red]")
        return None
    except Exception as e:
        console.print(f"[bold red]Error loading CSV file: {e}[/bold red]")
        return None

    df.columns = df.columns.str.strip()
    angle_col = next((col for col in df.columns if "b /" in col), "Angle")
    intensity_col = next((col for col in df.columns if "R_0" in col), "Intensity")
    df = df.rename(columns={angle_col: "Angle", intensity_col: "Intensity"})
    console.print(f"Successfully loaded data from [cyan]{input_path}[/cyan]")
    return df


def perform_peak_analysis(df: pd.DataFrame, params: dict, console: Console) -> dict:
    """Performs the core peak finding and fitting analysis."""
    console.print("\n[bold]--- Peak Analysis ---[/bold]")
    naive_params = {
        "threshold": params.get("threshold"),
        "distance": params.get("distance"),
        "prominence": params.get("prominence"),
        "width": params.get("width"),
    }
    initial_peaks_idx = find_all_peaks_naive(df, **naive_params)
    console.print(f"Found {len(initial_peaks_idx)} initial candidate peaks.")

    console.print("\n[bold]--- Fitting Global Background ---[/bold]")
    bg_params = fit_global_background(df, initial_peaks_idx, window=params["window"])
    if bg_params is None:
        console.print("[yellow]Background fitting failed. Using a zero background.[/yellow]")
        bg_params = (0, 0, 1)

    console.print("\n[bold]--- Fitting Peak Pairs ---[/bold]")
    all_fits = find_all_peaks_fitting(df, initial_peaks_idx, bg_params, window=params["window"])
    valid_fits = [fit for fit in all_fits if fit[1] is not None]
    console.print(f"Successfully fit {len(valid_fits)} out of {len(all_fits)} peak pairs.")

    return {
        "initial_peaks_idx": initial_peaks_idx,
        "bg_params": bg_params,
        "all_fits": all_fits,
        "valid_fits": valid_fits,
    }


def generate_summary_tables(
    df: pd.DataFrame, analysis_results: dict, wavelength: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates summary tables for peak details and d-spacing."""
    # Create Peak Details Table
    peak_table_data = []
    d_fitted_ka = []
    d_fitted_kb = []

    for i, (initial_idx, fit_params, _) in enumerate(analysis_results["all_fits"]):
        if fit_params is not None:
            mean_a = fit_params[1]
            d_a = bragg_d_spacing(mean_a, wavelength)
            d_fitted_ka.append(d_a)

            mean_b = np.rad2deg(2 * np.arcsin(np.sin(np.deg2rad(mean_a) / 2) * 0.9036))
            d_b = bragg_d_spacing(mean_b, wavelength)
            d_fitted_kb.append(d_b)

            peak_table_data.append(
                {
                    "Peak Pair": i + 1,
                    "K-a Angle": f"{mean_a:.4f}",
                    "K-a d (Å)": f"{d_a:.4f}",
                    "K-b Angle": f"{mean_b:.4f}",
                    "K-b d (Å)": f"{d_b:.4f}",
                }
            )
        else:
            initial_angle = df["Angle"].iloc[initial_idx]
            peak_table_data.append(
                {
                    "Peak Pair": i + 1,
                    "K-a Angle": f"Fit Failed @ {initial_angle:.2f}°",
                    "K-a d (Å)": "-",
                    "K-b Angle": "-",
                    "K-b d (Å)": "-",
                }
            )
    peak_df = pd.DataFrame(peak_table_data)

    # Create d-spacing Summary Table
    d_initial = [
        bragg_d_spacing(df["Angle"].iloc[i], wavelength)
        for i in analysis_results["initial_peaks_idx"]
    ]
    angles = df["Angle"].values
    y_total_fit = bremsstrahlung_bg(angles, *analysis_results["bg_params"])
    for _, fit_params, _ in analysis_results["valid_fits"]:
        y_total_fit += double_voigt(angles, *fit_params)
    final_model_peaks_idx, _ = find_peaks(y_total_fit, height=y_total_fit.max() * 0.05, distance=5)
    d_final_model = [bragg_d_spacing(angles[i], wavelength) for i in final_model_peaks_idx]

    sources = {
        "Initial (Naive) Peaks": d_initial,
        "Fitted K-alpha Peaks": d_fitted_ka,
        "Fitted K-beta Peaks": d_fitted_kb,
        "Final Model Peaks": d_final_model,
    }

    summary_data = []
    for name, d_list in sources.items():
        if d_list:
            result = find_most_probable_d(d_list)
            if result:
                mean_d, std_d, num_peaks = result
                summary_data.append(
                    {
                        "Data Source": name,
                        "Most Probable d (Å)": f"{mean_d:.4f}",
                        "Error (sigma)": f"{std_d:.4f}",
                        "Peaks Used": num_peaks,
                    }
                )
            else:
                summary_data.append(
                    {
                        "Data Source": name,
                        "Most Probable d (Å)": "Analysis Failed",
                        "Error (sigma)": "-",
                        "Peaks Used": len(d_list),
                    }
                )
        else:
            summary_data.append(
                {
                    "Data Source": name,
                    "Most Probable d (Å)": "-",
                    "Error (sigma)": "-",
                    "Peaks Used": 0,
                }
            )
    summary_df = pd.DataFrame(summary_data)

    return peak_df, summary_df
