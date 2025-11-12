from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from xray.bragg.calculations import (
    calculate_error_percentage,
    format_value_with_error,
    perform_bragg_fit_core,
    perform_combined_fit,
)
from xray.bragg.peak_finding import (
    find_all_peaks_fitting,
    find_all_peaks_naive,
    fit_global_background,
    fit_spectrum_with_predefined_peaks,
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


def perform_fitting_with_predefined_peaks(
    df: pd.DataFrame, predefined_angles: list[float], params: dict, console: Console
) -> dict:
    """Performs peak fitting using predefined angles as initial guesses."""
    console.print("Performing peak fitting with predefined angles...")

    # Convert predefined angles to indices
    angles_array = df["Angle"].values
    initial_peaks = [np.argmin(np.abs(angles_array - angle)) for angle in predefined_angles]
    initial_peaks = np.array(initial_peaks)

    # Empty properties dict as we are not running naive peak finding
    initial_peaks_properties = {}

    bg_params = fit_global_background(df, initial_peaks, window=params["window"])

    if bg_params is None:
        console.print("[bold red]Failed to fit global background.[/bold red]")
        return {}

    valid_fits = find_all_peaks_fitting(
        df, initial_peaks, initial_peaks_properties, bg_params, window=params["window"]
    )

    return {
        "initial_peaks_idx": initial_peaks,
        "initial_peaks_properties": initial_peaks_properties,
        "valid_fits": valid_fits,
        "bg_params": bg_params,
        "predefined_angles": predefined_angles,
    }


def perform_global_fit_analysis(
    df: pd.DataFrame, predefined_angles: list[float], console: Console
) -> dict:
    """Performs a global fit analysis on the data using predefined peaks."""
    console.print("Performing global fit analysis...")

    popt, _ = fit_spectrum_with_predefined_peaks(df, predefined_angles)

    if popt is None:
        console.print("[bold red]Global fit failed.[/bold red]")
        return {}

    bg_params = popt[0:3]
    voigt_params = popt[3:]
    num_peaks = len(predefined_angles)

    valid_fits = []
    initial_peaks_idx = []
    angles = df["Angle"].values

    for i in range(num_peaks):
        popt_peak = voigt_params[i * 4 : (i + 1) * 4]
        mean_angle = popt_peak[1]
        closest_idx = np.argmin(np.abs(angles - mean_angle))

        valid_fits.append((closest_idx, popt_peak, mean_angle))
        initial_peaks_idx.append(closest_idx)

    console.print("Global fit analysis completed.")

    return {
        "initial_peaks_idx": initial_peaks_idx,
        "initial_peaks_properties": {},
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
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Perform Bragg fit for given peak angles and single wavelength."""
    if not peak_angles:
        return np.array([]), np.array([]), np.nan, np.nan, np.nan

    # Prepare data for fit: single wavelength for all peaks
    data_with_wavelength = [
        (angle, wavelength, i) for i, angle in enumerate(np.sort(peak_angles), start=1)
    ]
    return perform_bragg_fit_core(data_with_wavelength)


def generate_summary_tables(
    df: pd.DataFrame,
    analysis_results: dict,
    wavelength: float,
    real_lattice_constant: float,
    predefined_angles: list[float],
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

    # Create processed peak data with predefined and fitted angles
    processed_peak_data = []
    for i, (idx, popt, fitted_mean_angle) in enumerate(analysis_results["valid_fits"]):
        predefined_mean_angle = predefined_angles[i] if i < len(predefined_angles) else np.nan
        if popt is not None:
            amplitude, sigma, gamma = popt[0], popt[2], popt[3]
        else:
            amplitude, sigma, gamma = np.nan, np.nan, np.nan
        processed_peak_data.append(
            {
                "Predefined Angle": predefined_mean_angle,
                "Fitted Angle": fitted_mean_angle,
                "Amplitude": amplitude,
                "Sigma": sigma,
                "Gamma": gamma,
            }
        )

    peak_df = pd.DataFrame(processed_peak_data)

    peak_df = peak_df.sort_values("Fitted Angle").reset_index(drop=True)

    # Now extract amplitudes, sigmas, gammas, and mean_angles from the sorted peak_df
    amplitudes = peak_df["Amplitude"].values
    sigmas = peak_df["Sigma"].values
    gammas = peak_df["Gamma"].values
    mean_angles = peak_df["Fitted Angle"].values

    # Identify K-alpha and K-beta peaks
    ka_peaks_angles, kb_peaks_angles, combined_data = identify_ka_kb_peaks(
        peak_df, analysis_results["valid_fits"]
    )

    # Perform fits for K-alpha, K-beta, and combined
    ka_sin_theta, ka_n_values, ka_slope, d_fit_ka, d_fit_ka_error = _perform_bragg_fit(
        ka_peaks_angles, lambda_a
    )
    kb_sin_theta, kb_n_values, kb_slope, d_fit_kb, d_fit_kb_error = _perform_bragg_fit(
        kb_peaks_angles, lambda_b
    )

    # Combined fit - uses both wavelengths appropriately for each point
    (
        combined_sin_theta,
        combined_n_values,
        combined_slope,
        d_fit_combined,
        d_fit_combined_error,
    ) = perform_combined_fit(combined_data, lambda_a, lambda_b)

    real_d_spacing = real_lattice_constant  # Assuming SC for d-spacing comparison

    error_d_ka = calculate_error_percentage(d_fit_ka, real_d_spacing)
    error_d_kb = calculate_error_percentage(d_fit_kb, real_d_spacing)
    error_d_combined = calculate_error_percentage(d_fit_combined, real_d_spacing)

    summary_df = pd.DataFrame(
        {
            "known_d_spacing (Angstrom)": [real_d_spacing],
            "inferred_ka_d_spacing (Angstrom)": [d_fit_ka],
            "inferred_ka_d_spacing_error (Angstrom)": [d_fit_ka_error],
            "error_ka_d_spacing (%)": [error_d_ka],
            "inferred_kb_d_spacing (Angstrom)": [d_fit_kb],
            "inferred_kb_d_spacing_error (Angstrom)": [d_fit_kb_error],
            "error_kb_d_spacing (%)": [error_d_kb],
            "inferred_combined_d_spacing (Angstrom)": [d_fit_combined],
            "inferred_combined_d_spacing_error (Angstrom)": [d_fit_combined_error],
            "error_combined_d_spacing (%)": [error_d_combined],
        }
    )

    fit_plot_data = {
        "ka_x_values": ka_sin_theta,
        "ka_y_values": ka_n_values,
        "ka_slope": ka_slope,
        "ka_d_fit": d_fit_ka,
        "ka_d_fit_error": d_fit_ka_error,
        "kb_x_values": kb_sin_theta,
        "kb_y_values": kb_n_values,
        "kb_slope": kb_slope,
        "kb_d_fit": d_fit_kb,
        "kb_d_fit_error": d_fit_kb_error,
        "combined_x_values": combined_sin_theta,
        "combined_y_values": combined_n_values,
        "combined_slope": combined_slope,
        "combined_d_fit": d_fit_combined,
        "combined_d_fit_error": d_fit_combined_error,
    }

    return peak_df, summary_df, fit_plot_data
