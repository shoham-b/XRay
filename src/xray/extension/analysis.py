
import pandas as pd
import numpy as np

from xray.path_manager import PathManager
from . import image_processing as ip
from . import calculations as calc
from . import viz
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def process_diffraction_image(image_path, phys_w_mm, phys_h_mm, distance_L_mm, wavelength_pm=71.1):
    """
    Analyzes X-ray film, generates plots, and saves results to a 'data' folder.
    """

    # --- 0. Setup Output Directory & Filenames ---
    path_manager = PathManager()
    base_output_dir = path_manager.get_artifacts_path() / "extension"
    
    graphs_dir = base_output_dir / "graphs"
    images_dir = base_output_dir / "images"
    
    graphs_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(image_path).stem

    logger.info(f"Processing: {image_path}")
    logger.info(f"Output directory: {base_output_dir.resolve()}")

    # --- 1. Load and Preprocess Image ---
    # Now returns 4 values: img_arr, img_inverted (array), img_no_blue_rgb (PIL), img_inverted_pil (PIL)
    img_arr, img_inverted, img_no_blue, img_inverted_pil = ip.load_and_preprocess_image(image_path)
    if img_arr is None:
        return

    # Save intermediate images
    img_no_blue.save(images_dir / f"{base_name}_preprocessed.png")
    img_inverted_pil.save(images_dir / f"{base_name}_inverted.png")

    h, w = img_arr.shape

    # --- 2. Find Center ---
    # User request: "remove the calibration cache"
    # Use automated center finding directly
    center_x, center_y, ring_radii, centers_dict = ip.find_center(img_inverted)
    logger.info(f"Center found at: ({center_x}, {center_y})")

    # --- 3. Calculate Radial Profile ---
    radial_profile = ip.calculate_radial_profile(img_inverted, center_x, center_y)

    # --- 4. Coordinate Conversion ---
    radii_mm, pixel_scale_mm = calc.pixels_to_mm(radial_profile, phys_w_mm, phys_h_mm, (h, w))
    two_theta_deg = calc.mm_to_2theta(radii_mm, distance_L_mm)

    # --- 5. Normalization ---
    profile_percent = calc.normalize_profile(radial_profile, pixel_scale_mm)
    
    # --- 6. Background Subtraction (Polynomial) ---
    # User request: "the fit is closse but not good enough, is there any other possible approach" -> ALS implemented.
    # User request: "it is much worse... bring back the cut of starting the fit data where drops below 99 percent of max intensity"
    # Reverting to Polynomial Fit with 99% threshold.
    # User request: "bring back my beloved 6 degree polinomial background fit"
    
    background_profile, start_idx = calc.fit_polynomial_background(radii_mm, profile_percent, saturation_threshold=90, degree=6)
    profile_subtracted = profile_percent - background_profile
    
    # Calculate start radius for visualization
    start_radius_mm = radii_mm[start_idx] if start_idx < len(radii_mm) else 0
    
    # Use subtracted profile for peak finding
    # User request: "apply smoothing before the peak finding"
    profile_for_peaks = calc.smooth_profile(profile_subtracted, window=11)
    
    # --- 7. Peak Finding ---
    # Find peaks in the subtracted data
    peak_indices, peak_properties, start_idx = calc.find_initial_peaks(profile_for_peaks, known_peak_indices=None)
    
    # We already have background and subtracted profile
    
    # --- 8. Sinc Fitting ---
    # Disabled per user request
    fitted_peaks = []

    # --- 8. Calculate d-spacings ---
    # Define peak_angles and peak_intensities here
    peak_angles = two_theta_deg[peak_indices]
    peak_intensities = profile_subtracted[peak_indices]
    d_spacings_pm = calc.calculate_d_spacings(peak_angles, wavelength_pm)
    
    # Calculate peak radii in pixels for visualization
    # peak_indices are indices into the radial_profile array, which corresponds to pixel radius
    peak_radii_pixels = peak_indices
    logger.info(f"Peak radii (pixels): {peak_radii_pixels}")
    if len(peak_radii_pixels) > 0:
        logger.info(f"  Min radius: {np.min(peak_radii_pixels)}")
        logger.info(f"  Max radius: {np.max(peak_radii_pixels)}")

    # --- 9. Plotting ---
    # Save center plot to 'images' directory
    # 1. With Pixels (Comparison)
    save_path_center_pixels = viz.plot_center_on_image(
        img_arr, center_x, center_y, peak_radii_pixels, base_name, images_dir, 
        centers_dict=centers_dict, filename_suffix="_center_pixels.png"
    )
    
    # 2. Clean (No Pixels)
    # Create a copy of centers_dict without the mask
    centers_dict_clean = centers_dict.copy()
    if 'ContourMask' in centers_dict_clean:
        del centers_dict_clean['ContourMask']
        
    save_path_center_clean = viz.plot_center_on_image(
        img_arr, center_x, center_y, peak_radii_pixels, base_name, images_dir, 
        centers_dict=centers_dict_clean, filename_suffix="_center_clean.png"
    )
    
    # Keep the original _center.png as the default (pixels) for backward compatibility/HTML
    # Keep the original _center.png as the default (pixels) for backward compatibility/HTML
    save_path_center = save_path_center_pixels
    
    # 3. Emboldened (Visualization)
    # User request: "expect the opposite of blurring to make it sharper"
    save_path_emboldened = viz.save_emboldened_image(img_inverted, base_name, images_dir)
    
    # 4. Rings (Visualization)
    # User request: "peaks are converted back to rings... plotted on top of the image"
    # User request: "The rings montage should also be on top of the blue change"
    # We use img_no_blue (PIL) converted to array
    img_no_blue_arr = np.array(img_no_blue)
    save_path_rings = viz.plot_rings_on_image(img_no_blue_arr, (center_x, center_y), peak_indices, base_name, images_dir)
    
    # --- 8. Calculate d-spacings for peaks ---
    peak_radii_mm = radii_mm[peak_indices]
    peak_two_theta = calc.mm_to_2theta(peak_radii_mm, distance_L_mm)
    peak_d_spacings = calc.calculate_d_spacings(peak_two_theta, wavelength_pm)

    # Save graphs to 'graphs' directory
    save_path_r = viz.plot_intensity_vs_radius(
        radii_mm, 
        profile_percent, 
        background_profile, 
        profile_subtracted, 
        base_name, 
        graphs_dir, 
        start_radius_mm=start_radius_mm,
        peak_radii=peak_radii_mm,
        peak_d_spacings=peak_d_spacings,
        profile_smoothed=profile_for_peaks
    )
    
    save_path_theta = viz.plot_intensity_vs_2theta(
        two_theta_deg, profile_percent, profile_subtracted,
        peak_angles, peak_intensities, d_spacings_pm,
        fitted_peaks, background_profile,
        base_name, distance_L_mm, graphs_dir, start_radius_mm=start_radius_mm,
        profile_smoothed=profile_for_peaks
    )

    # --- 10. Save Data to CSV ---
    df = pd.DataFrame({
        'Radius_mm': radii_mm,
        'TwoTheta_deg': two_theta_deg,
        'Intensity_Raw': radial_profile,
        'Intensity_Normalized_Percent': profile_percent,
        'Intensity_Background': background_profile,
        'Intensity_Subtracted': profile_subtracted,
        'Intensity_Smoothed': profile_for_peaks
    })

    csv_path = base_output_dir / f"{base_name}_analysis_data.csv"
    df.to_csv(csv_path, index=False)

    # --- 11. Generate Interactive Plots & HTML Report ---
    interactive_theta_fig = viz.create_interactive_plot(
        two_theta_deg, profile_percent, profile_subtracted,
        peak_angles, peak_intensities, d_spacings_pm,
        fitted_peaks, background_profile, base_name, start_radius_mm=start_radius_mm,
        profile_smoothed=profile_for_peaks
    )
    
    interactive_radius_fig = viz.create_interactive_radius_plot(
        radii_mm, 
        profile_percent, 
        background_profile, 
        profile_subtracted, 
        base_name,
        start_radius_mm=start_radius_mm,
        peak_radii=peak_radii_mm,
        peak_d_spacings=peak_d_spacings
    )
    
    save_path_preprocessed = images_dir / f"{base_name}_preprocessed.png"
    
    html_path = viz.generate_html_report(
        base_output_dir, base_name, save_path_r, save_path_theta, csv_path, 
        interactive_theta_fig, interactive_radius_fig, save_path_center,
        preprocessed_image_path=save_path_preprocessed
    )
    logger.info(f"2. {Path(save_path_r).as_uri()}")
    logger.info(f"3. {Path(save_path_theta).as_uri()}")
    logger.info(f"4. {Path(csv_path).as_uri()}")
    logger.info(f"5. {Path(html_path).as_uri()}")

    return {
        'base_name': base_name,
        'plot_r_path': save_path_r,
        'plot_theta_path': save_path_theta,
        'plot_center_path': save_path_center, # Default (pixels)
        'plot_center_pixels_path': save_path_center_pixels,
        'plot_center_clean_path': save_path_center_clean,
        'plot_emboldened_path': save_path_emboldened,
        'plot_rings_path': save_path_rings,
        'preprocessed_image_path': save_path_preprocessed,
        'csv_path': csv_path,
        'interactive_theta_fig': interactive_theta_fig,
        'interactive_radius_fig': interactive_radius_fig,
        'radii_mm': radii_mm,
        'profile_subtracted': profile_subtracted,
        'profile_smoothed': profile_for_peaks
    }
