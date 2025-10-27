from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from xray.analysis.peak_finding import (
    find_all_peaks_fitting,
    find_all_peaks_naive,
    fit_global_background,
)
from xray.mathutils import bragg_d_spacing
from xray.viz import plot_analysis_summary


def cli(
    input_file: Annotated[
        Path, typer.Option("--input", help="Input data file (CSV format).")
    ] = Path("data/dummy.csv"),
    data_dir: Annotated[
        Path | None,
        typer.Option("--data-dir", help="Base directory to resolve --input if it's relative."),
    ] = None,
    output_dir: Annotated[Path, typer.Option("--output", help="Directory to save plots.")] = Path(
        "artifacts"
    ),
    wavelength: Annotated[
        float, typer.Option("--wavelength", help="X-ray wavelength in Angstroms.")
    ] = 1.5406,  # Cu K-alpha,
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Relative height threshold (0-1) for peak detection."),
    ] = 0.1,
    distance: Annotated[
        int, typer.Option("--distance", help="Minimum number of points between peaks.")
    ] = 5,
    window: Annotated[
        int,
        typer.Option("--window", help="Half-window size in points used for local Voigt fitting."),
    ] = 20,
    prominence: Annotated[
        float | None,
        typer.Option("--prominence", help="Relative prominence (0-1) for peak detection."),
    ] = 0.1,
    width: Annotated[
        int | None, typer.Option("--width", help="Minimum peak width in points for detection.")
    ] = None,
) -> int:
    """Analyzes X-ray diffraction data to find peaks and calculate d-spacing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        input_path = (
            (data_dir / input_file)
            if (data_dir is not None and not input_file.is_absolute())
            else input_file
        )
        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            print(f"Warning: Failed to load CSV with UTF-8 ({e}), retrying with 'latin1' encoding.")
            df = pd.read_csv(input_path, encoding="latin1")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return 1
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return 1

    # Data Prep
    df.columns = df.columns.str.strip()
    angle_col = next((col for col in df.columns if "b /" in col), None)
    intensity_col = next((col for col in df.columns if "R_0" in col), None)
    if not angle_col or not intensity_col:
        angle_col = next((col for col in df.columns if "Angle" in col), "Angle")
        intensity_col = next((col for col in df.columns if "Intensity" in col), "Intensity")

    rename_map = {angle_col: "Angle", intensity_col: "Intensity"}
    df = df.rename(columns=rename_map)

    print(f"Successfully loaded data from {input_path}")

    # --- Peak Analysis ---
    print("\n--- Peak Analysis ---")
    initial_peaks = find_all_peaks_naive(
        df, threshold=threshold, distance=distance, prominence=prominence, width=width
    )
    print(f"Found {len(initial_peaks)} initial peaks.")

    print("\n--- Fitting Global Background ---")
    bg_params = fit_global_background(df, initial_peaks, window=window)
    if bg_params is None:
        print("Background fitting failed. Proceeding without background subtraction.")
        # Create a dummy background function that returns zero
        bg_params = (0, 0, 1)

    print("\n--- Fitting All Peaks (Double Voigt on Subtracted BG) ---")
    all_fits = find_all_peaks_fitting(df, initial_peaks, bg_params, window=window)

    valid_fits = [fit for fit in all_fits if fit[1] is not None]
    print(f"Successfully fit {len(valid_fits)} peaks out of {len(initial_peaks)} detected.")

    for i, (_peak_idx, fit_params, _fit_peak) in enumerate(valid_fits):
        if fit_params is not None:
            amp_a, mean_a, _, _, _ = fit_params
            d_spacing_fit = bragg_d_spacing(mean_a, wavelength)
            print(f"  Fit {i+1}: K-a at {mean_a:.4f}°, d={d_spacing_fit:.4f} Å")

    # --- Visualization ---
    print("\n--- Generating Plot ---")
    summary_plot_path = output_dir / f"{input_path.stem}_analysis_summary.png"
    try:
        plot_analysis_summary(df, initial_peaks, valid_fits, bg_params, summary_plot_path)
        print(f"Saved analysis summary plot to: {summary_plot_path}")
    except Exception as e:
        print(f"Failed to generate analysis summary plot: {e}")

    return 0


def main() -> None:
    """Entry point that runs the Typer-powered CLI."""
    typer.run(cli)


if __name__ == "__main__":
    main()
