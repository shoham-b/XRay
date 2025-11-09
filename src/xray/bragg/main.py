from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from xray.bragg.peak_finding import (
    find_all_peaks_fitting,
    find_all_peaks_naive,
    fit_global_background,
)


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


def generate_summary_tables(
        df: pd.DataFrame, analysis_results: dict, wavelength: float, real_lattice_constant: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Generates summary tables from the analysis results.
    1. Identifies K-alpha and K-beta pairs.
    2. Performs separate linear fits for K-alpha and K-beta peaks to find d-spacing.
    3. Performs a combined linear fit (K-alpha and normalized K-beta) to find d-spacing.
    4. Calculates potential lattice constants 'a' for cubic systems based on the fitted d values.
    5. Calculates error percentages for each inferred lattice constant compared to a real value.
    6. Returns dataframes and a dictionary with fit data for plotting.
    """

    if not analysis_results["valid_fits"]:
        return pd.DataFrame(), pd.DataFrame(), {}

    lambda_a = wavelength  # Assume the provided wavelength is K-alpha
    lambda_b = 0.6309  # Mo K-beta wavelength in Angstroms

    # Extract data from valid fits
    fit_data = []
    for _, popt, mean_angle in analysis_results["valid_fits"]:
        # Check if popt is a valid fit result or a placeholder
        if popt is not None and not (isinstance(popt, tuple) and all(p == 0 for p in popt)):
            # Use actual fit parameters
            fit_data.append((popt[0], popt[2], popt[3], mean_angle))
        else:
            # Use NaN for amplitude, sigma, gamma if no valid fit or if it's a placeholder
            fit_data.append((np.nan, np.nan, np.nan, mean_angle))

    if not fit_data:
        return pd.DataFrame(), pd.DataFrame(), {}

    amplitudes, sigmas, gammas, mean_angles = zip(*fit_data, strict=False)

    # Create a dataframe to work with the peaks
    peaks_data = (
        pd.DataFrame(
            {"amplitude": amplitudes, "sigma": sigmas, "gamma": gammas, "angle": mean_angles}
        )
        .sort_values("angle")
        .reset_index(drop=True)
    )

    ka_peaks_angles = []
    kb_peaks_angles = []
    combined_sin_thetas_for_fit = []  # For the combined fit

    # Check if amplitudes are all NaN, indicating predefined angles were used
    if peaks_data["amplitude"].isnull().all():
        # If all amplitudes are NaN, we use the original index from valid_fits to determine K-alpha/K-beta
        for i, (_, _, mean_angle) in enumerate(analysis_results["valid_fits"]):
            if i % 2 != 0:  # Odd indices are K-alpha
                ka_peaks_angles.append(mean_angle)
                combined_sin_thetas_for_fit.append(np.sin(np.deg2rad(mean_angle / 2)))
            else:  # Even indices are K-beta
                kb_peaks_angles.append(mean_angle)
                sin_theta_b = np.sin(np.deg2rad(mean_angle / 2))
                sin_theta_a_equiv = (lambda_a / lambda_b) * sin_theta_b
                combined_sin_thetas_for_fit.append(sin_theta_a_equiv)
    else:
        for i in range(0, len(peaks_data), 2):
            if i + 1 < len(peaks_data):
                pair = peaks_data.iloc[i: i + 2]
                # Original logic for identifying K-alpha and K-beta based on amplitude
                ka_peak = pair.loc[pair["amplitude"].idxmax()]
                kb_peak = pair.loc[pair["amplitude"].idxmin()]

                ka_peaks_angles.append(ka_peak["angle"])
                kb_peaks_angles.append(kb_peak["angle"])

                # For combined fit: add K-alpha sin(theta)
                combined_sin_thetas_for_fit.append(np.sin(np.deg2rad(ka_peak["angle"] / 2)))
                # For combined fit: add normalized K-beta sin(theta)
                sin_theta_b = np.sin(np.deg2rad(kb_peak["angle"] / 2))
                sin_theta_a_equiv = (lambda_a / lambda_b) * sin_theta_b
                combined_sin_thetas_for_fit.append(sin_theta_a_equiv)
            else:
                # Handle lone peak (assume it's K-alpha)
                lone_peak = peaks_data.iloc[i]
                ka_peaks_angles.append(lone_peak["angle"])
                combined_sin_thetas_for_fit.append(np.sin(np.deg2rad(lone_peak["angle"] / 2)))

    # --- K-alpha only fit ---
    ka_thetas_sorted = np.deg2rad(np.sort(ka_peaks_angles) / 2)
    ka_sin_theta_sorted = np.sin(ka_thetas_sorted)
    ka_n_values = np.arange(1, len(ka_sin_theta_sorted) + 1)

    ka_m_prime = np.sum(ka_sin_theta_sorted * ka_n_values) / np.sum(
        ka_sin_theta_sorted * ka_sin_theta_sorted
    )
    d_fit_ka = ka_m_prime * lambda_a / 2

    # --- K-beta only fit ---
    kb_thetas_sorted = np.deg2rad(np.sort(kb_peaks_angles) / 2)
    kb_sin_theta_sorted = np.sin(kb_thetas_sorted)
    kb_n_values = np.arange(1, len(kb_sin_theta_sorted) + 1)

    kb_m_prime = np.sum(kb_sin_theta_sorted * kb_n_values) / np.sum(
        kb_sin_theta_sorted * kb_sin_theta_sorted
    )
    d_fit_kb = kb_m_prime * lambda_b / 2

    # --- Combined fit (K-alpha and normalized K-beta) ---
    combined_sin_theta_sorted = np.array(sorted(combined_sin_thetas_for_fit))
    combined_n_values = np.arange(1, len(combined_sin_theta_sorted) + 1)

    combined_m_prime = np.sum(combined_sin_theta_sorted * combined_n_values) / np.sum(
        combined_sin_theta_sorted * combined_sin_theta_sorted
    )
    d_fit_combined = combined_m_prime * lambda_a / 2  # Use K-alpha wavelength for combined fit

    # Calculate potential 'a' values for cubic lattices
    a_sc_ka = d_fit_ka * np.sqrt(1)
    a_bcc_ka = d_fit_ka * np.sqrt(2)
    a_fcc_ka = d_fit_ka * np.sqrt(3)

    a_sc_kb = d_fit_kb * np.sqrt(1) if kb_peaks_angles else np.nan
    a_bcc_kb = d_fit_kb * np.sqrt(2) if kb_peaks_angles else np.nan
    a_fcc_kb = d_fit_kb * np.sqrt(3) if kb_peaks_angles else np.nan

    a_sc_combined = d_fit_combined * np.sqrt(1)
    a_bcc_combined = d_fit_combined * np.sqrt(2)
    a_fcc_combined = d_fit_combined * np.sqrt(3)

    # Calculate error percentages
    def calculate_error_percentage(inferred_a, real_a):
        if np.isnan(inferred_a) or real_a == 0:
            return np.nan
        return ((inferred_a - real_a) / real_a) * 100

    error_sc_ka = calculate_error_percentage(a_sc_ka, real_lattice_constant)
    error_bcc_ka = calculate_error_percentage(a_bcc_ka, real_lattice_constant)
    error_fcc_ka = calculate_error_percentage(a_fcc_ka, real_lattice_constant)

    error_sc_kb = calculate_error_percentage(a_sc_kb, real_lattice_constant)
    error_bcc_kb = calculate_error_percentage(a_bcc_kb, real_lattice_constant)
    error_fcc_kb = calculate_error_percentage(a_fcc_kb, real_lattice_constant)

    error_sc_combined = calculate_error_percentage(a_sc_combined, real_lattice_constant)
    error_bcc_combined = calculate_error_percentage(a_bcc_combined, real_lattice_constant)
    error_fcc_combined = calculate_error_percentage(a_fcc_combined, real_lattice_constant)

    peak_df = pd.DataFrame(
        {
            "Angle": mean_angles,
            "Amplitude": amplitudes,
            "Sigma": sigmas,
            "Gamma": gammas,
        }
    )

    summary_data = {
        "inferred_ka_d_spacing": [d_fit_ka],
        "inferred_kb_d_spacing": [d_fit_kb],
        "inferred_combined_d_spacing": [d_fit_combined],
        "a_SC_ka": [a_sc_ka],
        "a_BCC_ka": [a_bcc_ka],
        "a_FCC_ka": [a_fcc_ka],
        "error_SC_ka": [error_sc_ka],
        "error_BCC_ka": [error_bcc_ka],
        "error_FCC_ka": [error_fcc_ka],
        "a_SC_kb": [a_sc_kb],
        "a_BCC_kb": [a_bcc_kb],
        "a_FCC_kb": [a_fcc_kb],
        "error_SC_kb": [error_sc_kb],
        "error_BCC_kb": [error_bcc_kb],
        "error_FCC_kb": [error_fcc_kb],
        "a_SC_combined": [a_sc_combined],
        "a_BCC_combined": [a_bcc_combined],
        "a_FCC_combined": [a_fcc_combined],
        "error_SC_combined": [error_sc_combined],
        "error_BCC_combined": [error_bcc_combined],
        "error_FCC_combined": [error_fcc_combined],
    }

    summary_df = pd.DataFrame(summary_data)

    fit_plot_data = {
        "ka_x_values": ka_sin_theta_sorted,
        "ka_y_values": ka_n_values,
        "ka_slope": ka_m_prime,
        "kb_x_values": kb_sin_theta_sorted,
        "kb_y_values": kb_n_values,
        "kb_slope": kb_m_prime,
        "combined_x_values": combined_sin_theta_sorted,
        "combined_y_values": combined_n_values,
        "combined_slope": combined_m_prime,
    }

    return peak_df, summary_df, fit_plot_data


