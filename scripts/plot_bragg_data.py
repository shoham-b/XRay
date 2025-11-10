from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from xray.bragg.main import generate_summary_tables
from xray.bragg.peak_finding import (
    find_all_peaks_fitting,
    find_all_peaks_naive,
    fit_global_background,
)
from xray.bragg.viz import create_multi_material_report

K_ALPHA_WAVELENGTH_MO = 0.7093  # Angstroms
REAL_LATTICE_CONSTANT_NACL = 5.64  # Angstroms
REAL_LATTICE_CONSTANT_LIF = 4.03  # Angstroms


def main():
    console = Console()

    # Read the CSV file
    data_file = Path("data/bragg/NaCl.csv")  # Using the correct filename
    material_name = "NaCl" if "NaCl" in str(data_file) else "LiF"

    # Determine the real lattice constant based on the material
    if material_name == "NaCl":
        real_lattice_constant = REAL_LATTICE_CONSTANT_NACL
    elif material_name == "LiF":
        real_lattice_constant = REAL_LATTICE_CONSTANT_LIF
    else:
        console.print(
            f"[bold red]Warning: Unknown material '{material_name}'. Using default real lattice constant for NaCl.[/bold red]"
        )
        real_lattice_constant = REAL_LATTICE_CONSTANT_NACL

    console.print(f"Processing {material_name} data...")

    # Read and verify the data
    console.print(f"Reading data from: {data_file}")
    df = pd.read_csv(data_file, sep=",", skipinitialspace=True)
    df.columns = ["Angle", "Intensity"]  # Explicitly set column names
    console.print(f"Raw data shape: {df.shape}")

    # Ensure Angle and Intensity columns are numeric
    df["Angle"] = pd.to_numeric(df["Angle"])
    df["Intensity"] = pd.to_numeric(df["Intensity"])

    console.print(f"Data range - Angle: {df['Angle'].min():.2f} to {df['Angle'].max():.2f}")
    console.print(f"Intensity range: {df['Intensity'].min():.2f} to {df['Intensity'].max():.2f}")

    # Preprocess the data
    # 1. Normalize intensity to 0-1 range
    min_intensity = df["Intensity"].min()
    max_intensity = df["Intensity"].max()
    df["Intensity"] = (df["Intensity"] - min_intensity) / (max_intensity - min_intensity)

    # 2. Smooth the data using a small window
    df["Intensity"] = df["Intensity"].rolling(window=3, center=True, min_periods=1).median()

    # 3. Calculate rolling statistics for adaptive thresholding
    window_size = 50  # Adjust based on your data density
    df["rolling_median"] = (
        df["Intensity"].rolling(window=window_size, center=True, min_periods=1).median()
    )
    df["rolling_std"] = (
        df["Intensity"].rolling(window=window_size, center=True, min_periods=1).std()
    )

    # Parameters for peak detection - more sensitive settings
    params = {
        "threshold": 0.03,  # Very low threshold to detect weak peaks
        "distance": 3,  # Minimum distance between peaks (in data points)
        "prominence": 0.05,  # Reduced prominence for weaker peaks
        "width": 1,  # Minimum peak width
        "window": 15,  # Smaller window for better local fitting
        "rel_height": 0.5,  # For peak width calculation
        "wlen": None,  # Window length for prominence calculation (None = full data)
    }

    # First pass: Find prominent peaks
    console.print("\nStarting peak detection...")
    try:
        initial_peaks, initial_peaks_properties = find_all_peaks_naive(
            df,
            threshold=params["threshold"],
            distance=params["distance"],
            prominence=params["prominence"],
        )
        console.print(f"First pass found {len(initial_peaks)} peaks")
        if len(initial_peaks) > 0:
            angles = df["Angle"].iloc[initial_peaks].values
            console.print(f"Peak angles (2θ): {', '.join(f'{a:.2f}°' for a in angles)}")
    except Exception as e:
        console.print(f"Error in first pass peak detection: {e}")
        initial_peaks = np.array([])
        initial_peaks_properties = {}

    # If too few peaks found, try with more sensitive settings
    if len(initial_peaks) < 8:  # Expecting at least 8 peaks for NaCl
        console.print("First pass found too few peaks. Trying with more sensitive settings...")
        initial_peaks, initial_peaks_properties = find_all_peaks_naive(
            df,
            threshold=0.02,  # Even lower threshold
            distance=2,  # Allow peaks closer together
            prominence=0.03,  # Lower prominence threshold
            width=1,  # Minimum width
        )

    console.print(f"Initial peak detection found {len(initial_peaks)} peaks")

    # Fit background with a larger window to better capture the baseline
    bg_window = min(30, len(df) // 10)  # Use 10% of data points or 30, whichever is smaller
    bg_params = fit_global_background(df, initial_peaks, window=bg_window)

    # If background fitting failed, use a simple linear background
    if bg_params is None:
        console.print("Warning: Background fitting failed, using linear approximation")
        x = df["Angle"].values
        y = df["Intensity"].values
        bg_params = (0.0, x[0], (y[-1] - y[0]) / (x[-1] - x[0]))

    # Find all peaks with proper fitting
    all_fits = find_all_peaks_fitting(
        df, initial_peaks, initial_peaks_properties, bg_params, window=params["window"]
    )

    # Filter out None fits and sort by angle
    valid_fits = [(i, popt, angle) for i, popt, angle in all_fits if popt is not None]
    valid_fits.sort(key=lambda x: x[2])  # Sort by angle

    # Get the peak indices for visualization
    np.array([i for i, _, _ in valid_fits])

    console.print(f"Found {len(valid_fits)} valid peaks for {material_name}")

    # Generate summary tables and fit plot data
    peak_table, summary_table, fit_plot_data = generate_summary_tables(
        df,
        {
            "initial_peaks_idx": initial_peaks,
            "initial_peaks_properties": initial_peaks_properties,
            "valid_fits": valid_fits,
            "bg_params": bg_params,
        },
        wavelength=K_ALPHA_WAVELENGTH_MO,
        real_lattice_constant=real_lattice_constant,
    )

    # Create output directory if it doesn't exist
    out_dir = Path("artifacts/bragg_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate the interactive report
    output_path = out_dir / "bragg_analysis_report.html"
    create_multi_material_report(
        analysis_data_list=[
            {
                "name": material_name,
                "df": df,
                "analysis_results": {
                    "initial_peaks_idx": initial_peaks,
                    "valid_fits": valid_fits,
                    "bg_params": bg_params,
                },
                "peak_df": peak_table,
                "summary_df": summary_table,
                "fit_plot_data": fit_plot_data,
                "real_lattice_constant": real_lattice_constant,
            }
        ],
        out_path=output_path,
    )

    print(f"Report generated successfully at: {output_path}")


if __name__ == "__main__":
    main()
