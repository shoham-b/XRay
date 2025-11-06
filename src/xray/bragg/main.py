from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from xray.bragg.hkl import find_hkl
from xray.bragg.image_processing import (
    find_big_circle,
    find_small_dots,
    load_and_preprocess_image,
    save_dots_to_csv,
    visualize_and_save_results,
)
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

    initial_peaks = find_all_peaks_naive(
        df,
        threshold=params["threshold"],
        distance=params["distance"],
        prominence=params["prominence"],
        width=params["width"],
    )

    bg_params = fit_global_background(df, initial_peaks, window=params["window"])

    if bg_params is None:
        console.print("[bold red]Failed to fit global background.[/bold red]")
        return {}

    valid_fits = find_all_peaks_fitting(df, initial_peaks, bg_params, window=params["window"])

    console.print("Peak analysis completed.")

    return {
        "initial_peaks_idx": initial_peaks,
        "valid_fits": valid_fits,
        "bg_params": bg_params,
    }


def generate_summary_tables(
    df: pd.DataFrame, analysis_results: dict, wavelength: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates summary tables from the analysis results."""

    peak_data = []
    for _, popt, mean_angle in analysis_results["valid_fits"]:
        if popt is not None:
            d_spacing = wavelength / (2 * np.sin(np.deg2rad(mean_angle / 2)))
            peak_data.append([mean_angle, popt[0], popt[2], popt[3], d_spacing])

    peak_df = pd.DataFrame(peak_data, columns=["Angle", "Amplitude", "Sigma", "Gamma", "d-spacing"])

    summary_df = peak_df.agg({"d-spacing": ["mean", "median", "std"]}).reset_index()

    return peak_df, summary_df


def run_bragg_analysis(
    image_path,
    output_dir,
    big_circle_thresh,
    small_dot_thresh,
    min_spot_area,
    min_circularity,
    phys_y_mm,
    phys_x_mm,
    l_mm,
    a_0_pm,
    small_dot_thresh_outer,
    max_distance_percentage,
):
    # --- 1. Load Image and Pre-process ---
    image, output_image, blurred_v, px_height, px_width = load_and_preprocess_image(image_path)
    if image is None:
        return

    # --- 2. Find the Big Circle ---
    c_big_circle, big_circle_center, thresh_big_circle = find_big_circle(
        blurred_v, big_circle_thresh
    )
    if c_big_circle is None:
        return

    # --- 3. Find the Small Dots ---
    detected_dots, thresh_small_dots, rejected_dots_circularity = find_small_dots(
        blurred_v,
        small_dot_thresh,
        min_spot_area,
        min_circularity,
        big_circle_center=big_circle_center,
        c_big_circle=c_big_circle,
        px_height=px_height,
        px_width=px_width,
        small_dot_thresh_outer=small_dot_thresh_outer,
        max_distance_percentage=max_distance_percentage,
    )
    if not detected_dots:
        return

    # --- 4. Print Final Results (Centers) ---
    print("\n--- FINAL RESULTS (Pixel Coordinates) ---")
    print(f"Center of Big Circle: {big_circle_center}")
    detected_dots.sort(key=lambda d: (d["center"][1], d["center"][0]))
    for i, dot in enumerate(detected_dots):
        print(f"  Dot {i + 1} center: {dot['center']}")

    # --- 5. CALCULATIONS (x, y, z, h, k, l, d, lambda) ---
    print("\n--- CALCULATIONS (d and lambda) ---")

    mm_per_px_x = phys_x_mm / px_width
    mm_per_px_y = phys_y_mm / px_height

    print(f"Conversion: {mm_per_px_x:.4f} mm/px (X), {mm_per_px_y:.4f} mm/px (Y)")
    print(f"Assuming Lattice Constant a_0 = {a_0_pm} pm (NaCl)\n")

    # Print new table header
    print(
        f"{'Dot':<4} | {'(x_Q mm)':<10} | {'(y_Q mm)':<10} | {'(z_Q mm)':<10} | "
        f"{'h, k, ell':<12} | {'d (pm)':<10} | {'Theta (deg)':<12} | {'lambda (pm)':<10}"
    )
    print("-" * 97)

    for i, dot in enumerate(detected_dots):
        center_px = dot["center"]
        dx_px = center_px[0] - big_circle_center[0]
        dy_px = center_px[1] - big_circle_center[1]

        x_mm = dx_px * mm_per_px_x
        y_mm = dy_px * mm_per_px_y
        z_mm = l_mm  # z_Q is the constant sample-to-film distance

        h, k, ell = find_hkl(x_mm, y_mm, z_mm)
        dot["hkl"] = (h, k, ell)  # Store for plotting
        dot["x_mm"] = x_mm
        dot["y_mm"] = y_mm
        dot["z_mm"] = z_mm

        # --- New Calculations for d and lambda ---
        hkl_norm_sq = float(h**2 + k**2 + ell**2)

        if hkl_norm_sq == 0:
            d_pm = np.inf
            lambda_pm = np.nan
            theta_rad = 0.0
        else:
            # 1. Calculate d
            hkl_norm = np.sqrt(hkl_norm_sq)
            d_pm = a_0_pm / hkl_norm

            # 2. Calculate Bragg angle theta (vartheta)
            theta_rad = np.pi / 2.0 if ell == 0 else np.arctan(np.sqrt(h**2 + k**2) / ell)

            # 3. Calculate lambda
            lambda_pm = 2 * d_pm * np.sin(theta_rad)

        dot["d_pm"] = d_pm
        dot["theta_deg"] = np.degrees(theta_rad)
        dot["lambda_pm"] = lambda_pm

        # Convert theta to degrees for printing
        theta_deg = np.degrees(theta_rad)

        # Print all values in the new table
        print(
            f"{(i + 1):<4} | {x_mm:<10.3f} | {y_mm:<10.3f} | {z_mm:<10.1f} | "
            f"{(h, k, ell)!s:<12} | {d_pm:<10.2f} | {theta_deg:<12.3f} | "
            f"{lambda_pm:<10.2f}"
        )

    # --- 6. Visualize and Save Results ---
    visualize_and_save_results(
        image,
        output_image,
        c_big_circle,
        big_circle_center,
        detected_dots,
        thresh_big_circle,
        thresh_small_dots,
        output_dir,
        rejected_dots_circularity=rejected_dots_circularity,
    )

    # --- 7. Save Results to CSV ---
    save_dots_to_csv(detected_dots, output_dir)
