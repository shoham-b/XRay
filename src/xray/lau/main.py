from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from scipy.signal import find_peaks

from xray.lau.hkl import find_hkl
from xray.lau.image_processing import (
    find_big_circle,
    find_small_dots,
    load_and_preprocess_image,
    save_dots_to_csv,
    visualize_and_save_results,
)
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


def run_lau_analysis(
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
            f"{'h, k, ell':<12} | {d_pm:<10.2f} | {theta_deg:<12.3f} | "
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
