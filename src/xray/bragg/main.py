import numpy as np

from xray.bragg.hkl import find_hkl
from xray.bragg.image_processing import (
    find_big_circle,
    find_small_dots,
    load_and_preprocess_image,
    save_dots_to_csv,
    visualize_and_save_results,
)


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
