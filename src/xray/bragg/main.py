from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from xray.bragg.calculations import (
    calculate_cubic_lattice_constants,
    calculate_d_spacing,
    calculate_error_percentage,
    perform_bragg_fit_core,
    perform_combined_fit,
    perform_linear_fit,
)
from xray.bragg.peak_finding import (
    find_all_peaks_fitting,
    find_all_peaks_naive,
    fit_global_background,
)
from xray.bragg.peak_pairing import identify_ka_kb_peaks


def load_and_prep_data(input_path: Path, console: Console) -> pd.DataFrame | None:
    """Loads and prepares the data from a CSV file."""
    try:
        console.print(f"Loading data from [cyan]{input_path}[/cyan]")
        df = pd.read_csv(input_path)
        df.columns = ["Angle", "Intensity"]
        df["Intensity"] = df["Intensity"].rolling(window=5, center=True).mean()
        df = df.dropna()
        console.print("Data loaded and preprocessed successfully.")
        return df
    except FileNotFoundError:
        console.print(f"[bold red]Error: File not found at {input_path}[/bold red]")
        return None
    except Exception as e:
        console.print(f"[bold red]An error occurred while loading the data: {e}[/bold red]")
        return None


def perform_peak_analysis(df: pd.DataFrame, params: dict, console: Console) -> dict:
    """Performs peak analysis on the data."""
    console.print("Performing peak analysis...")

    initial_peaks, initial_peaks_properties = find_all_peaks_naive(
        df,
        threshold=params["threshold"],
        distance=params["distance"],
        prominence=params["prominence"],
    )

    bg_params = fit_global_background(df, initial_peaks, window=params["window"])

    if bg_params is None:
        console.print("[bold red]Failed to fit global background.[/bold red]")
        return {}

    valid_fits = find_all_peaks_fitting(
        df, initial_peaks, initial_peaks_properties, bg_params, window=params["window"]
    )

    console.print("Peak analysis completed.")

    return {
        "initial_peaks_idx": initial_peaks,
        "initial_peaks_properties": initial_peaks_properties,
        "valid_fits": valid_fits,
        "bg_params": bg_params,
    }


def _extract_fit_data(valid_fits: list) -> list[tuple]:
    """Extract fit data from valid fits results."""
    fit_data = []
    for _, popt, mean_angle in valid_fits:
        if popt is not None and not (isinstance(popt, tuple) and all(p == 0 for p in popt)):
            fit_data.append((popt[0], popt[2], popt[3], mean_angle))
        else:
            fit_data.append((np.nan, np.nan, np.nan, mean_angle))
    return fit_data


def _create_peaks_dataframe(fit_data: list[tuple]) -> pd.DataFrame:
    """Create sorted DataFrame from fit data."""
    amplitudes, sigmas, gammas, mean_angles = zip(*fit_data, strict=False)
    return (
        pd.DataFrame(
            {"amplitude": amplitudes, "sigma": sigmas, "gamma": gammas, "angle": mean_angles}
        )
        .sort_values("angle")
        .reset_index(drop=True)
    )


def _perform_bragg_fit(
    peak_angles: list[float], wavelength: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Perform Bragg fit for given peak angles and single wavelength."""
    if not peak_angles:
        return np.array([]), np.array([]), np.nan, np.nan
    
    # Prepare data for fit: single wavelength for all peaks
    data_with_wavelength = [(angle, wavelength, i) for i, angle in enumerate(np.sort(peak_angles), start=1)]
    return perform_bragg_fit_core(data_with_wavelength)


def generate_summary_tables(
    df: pd.DataFrame, analysis_results: dict, wavelength: float, real_lattice_constant: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Generates summary tables from the analysis results.
    
    1. Identifies K-alpha and K-beta pairs.
    2. Performs separate linear fits for K-alpha and K-beta peaks to find d-spacing.
    3. Performs a combined linear fit to find d-spacing.
    4. Calculates potential lattice constants for cubic systems.
    5. Calculates error percentages compared to known values.
    
    Returns:
        Tuple of (peak_df, summary_df, fit_plot_data)
    """
    if not analysis_results["valid_fits"]:
        return pd.DataFrame(), pd.DataFrame(), {}

    lambda_a = wavelength  # K-alpha wavelength
    lambda_b = 0.6309  # Mo K-beta wavelength in Angstroms

    # Extract and prepare data
    fit_data = _extract_fit_data(analysis_results["valid_fits"])
    if not fit_data:
        return pd.DataFrame(), pd.DataFrame(), {}

    peaks_data = _create_peaks_dataframe(fit_data)
    amplitudes, sigmas, gammas, mean_angles = zip(*fit_data, strict=False)

    # Identify K-alpha and K-beta peaks
    ka_peaks_angles, kb_peaks_angles, combined_data = identify_ka_kb_peaks(
        peaks_data, analysis_results["valid_fits"]
    )

    # Perform fits for K-alpha, K-beta, and combined
    ka_sin_theta, ka_n_values, ka_slope, d_fit_ka = _perform_bragg_fit(ka_peaks_angles, lambda_a)
    kb_sin_theta, kb_n_values, kb_slope, d_fit_kb = _perform_bragg_fit(kb_peaks_angles, lambda_b)
    
    # Combined fit - uses both wavelengths appropriately for each point
    combined_sin_theta, combined_n_values, combined_slope, d_fit_combined = perform_combined_fit(
        combined_data, lambda_a, lambda_b
    )

    # Calculate lattice constants
    lattice_ka = calculate_cubic_lattice_constants(d_fit_ka)
    lattice_kb = calculate_cubic_lattice_constants(d_fit_kb)
    lattice_combined = calculate_cubic_lattice_constants(d_fit_combined)

    # Calculate errors
    errors_ka = {
        k: calculate_error_percentage(v, real_lattice_constant)
        for k, v in lattice_ka.items()
    }
    errors_kb = {
        k: calculate_error_percentage(v, real_lattice_constant)
        for k, v in lattice_kb.items()
    }
    errors_combined = {
        k: calculate_error_percentage(v, real_lattice_constant)
        for k, v in lattice_combined.items()
    }

    # Create output DataFrames
    peak_df = pd.DataFrame(
        {
            "Angle": mean_angles,
            "Amplitude": amplitudes,
            "Sigma": sigmas,
            "Gamma": gammas,
        }
    )

    summary_df = pd.DataFrame({
        "inferred_ka_d_spacing": [d_fit_ka],
        "inferred_kb_d_spacing": [d_fit_kb],
        "inferred_combined_d_spacing": [d_fit_combined],
        "a_SC_ka": [lattice_ka["sc"]],
        "a_BCC_ka": [lattice_ka["bcc"]],
        "a_FCC_ka": [lattice_ka["fcc"]],
        "error_SC_ka": [errors_ka["sc"]],
        "error_BCC_ka": [errors_ka["bcc"]],
        "error_FCC_ka": [errors_ka["fcc"]],
        "a_SC_kb": [lattice_kb["sc"]],
        "a_BCC_kb": [lattice_kb["bcc"]],
        "a_FCC_kb": [lattice_kb["fcc"]],
        "error_SC_kb": [errors_kb["sc"]],
        "error_BCC_kb": [errors_kb["bcc"]],
        "error_FCC_kb": [errors_kb["fcc"]],
        "a_SC_combined": [lattice_combined["sc"]],
        "a_BCC_combined": [lattice_combined["bcc"]],
        "a_FCC_combined": [lattice_combined["fcc"]],
        "error_SC_combined": [errors_combined["sc"]],
        "error_BCC_combined": [errors_combined["bcc"]],
        "error_FCC_combined": [errors_combined["fcc"]],
    })

    fit_plot_data = {
        "ka_x_values": ka_sin_theta,
        "ka_y_values": ka_n_values,
        "ka_slope": ka_slope,
        "kb_x_values": kb_sin_theta,
        "kb_y_values": kb_n_values,
        "kb_slope": kb_slope,
        "combined_x_values": combined_sin_theta,
        "combined_y_values": combined_n_values,
        "combined_slope": combined_slope,
    }

    return peak_df, summary_df, fit_plot_data


