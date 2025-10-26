from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from xray.analysis.peak_finding import find_peaks_fitting, find_peaks_naive
from xray.mathutils import bragg_d_spacing
from xray.viz import plot_peak_fit, plot_scan


def cli(
    input_file: Annotated[Path, typer.Option("--input", help="Input data file (Excel format).")],
    output_dir: Annotated[Path, typer.Option("--output", help="Directory to save plots.")] = Path(
        "artifacts"
    ),
    wavelength: Annotated[
        float, typer.Option("--wavelength", help="X-ray wavelength in Angstroms.")
    ] = 1.5406,  # Cu K-alpha
) -> int:
    """Analyzes X-ray diffraction data to find peaks and calculate d-spacing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        # The data manager is instantiated without arguments, assuming default paths.
        # For a CLI, it's better to work with the direct file path provided.
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return 1
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    print(f"Successfully loaded data from {input_file}")

    # --- Peak Analysis ---
    print("\n--- Peak Analysis ---")

    # 1. Naive Peak Finding
    naive_peak = find_peaks_naive(df)
    if not naive_peak.empty:
        angle_naive = naive_peak["Angle"].iloc[0]
        intensity_naive = naive_peak["Intensity"].iloc[0]
        d_spacing_naive = bragg_d_spacing(angle_naive, wavelength)
        print(f"Naive Peak found at: Angle = {angle_naive:.4f}°, Intensity = {intensity_naive:.2f}")
        print(f"  - Calculated d-spacing: {d_spacing_naive:.4f} Å")

    # 2. Peak Fitting
    try:
        fit_params = find_peaks_fitting(df)
        amplitude, mean_angle, stddev = fit_params
        d_spacing_fit = bragg_d_spacing(mean_angle, wavelength)
        print(f"Fitted Peak found at: Angle = {mean_angle:.4f}0")
        print(
            f"  - Fit parameters: Amplitude={amplitude:.2f}, "
            f"Mean={mean_angle:.4f}, StdDev={stddev:.4f}"
        )
        print(f"  - Calculated d-spacing: {d_spacing_fit:.4f} 5")
    except Exception as e:
        print(f"Peak fitting failed: {e}")
        fit_params = None

    # --- Visualization ---
    print("\n--- Generating Plots ---")

    # 1. Raw Scan Plot
    scan_plot_path = output_dir / f"{input_file.stem}_scan.png"
    try:
        plot_scan(df, scan_plot_path)
        print(f"Saved scan plot to: {scan_plot_path}")
    except Exception as e:
        print(f"Failed to generate scan plot: {e}")

    # 2. Peak Fit Plot
    if fit_params:
        fit_plot_path = output_dir / f"{input_file.stem}_fit.png"
        try:
            plot_peak_fit(df, fit_params, fit_plot_path)
            print(f"Saved peak fit plot to: {fit_plot_path}")
        except Exception as e:
            print(f"Failed to generate fit plot: {e}")

    return 0


def main() -> None:
    """Entry point that runs the Typer-powered CLI."""
    typer.run(cli)


if __name__ == "__main__":
    main()
