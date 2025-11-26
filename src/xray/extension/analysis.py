import os
import pandas as pd
import numpy as np

from xray.path_manager import PathManager
from . import image_processing as ip
from . import calculations as calc
from . import viz
from pathlib import Path
def process_diffraction_image(image_path, phys_w_mm, phys_h_mm, distance_L_mm, wavelength_pm=71.1):
    """
    Analyzes X-ray film, generates plots, and saves results to a 'data' folder.
    """

    # --- 0. Setup Output Directory & Filenames ---
    path_manager = PathManager()
    base_output_dir = path_manager.get_artifacts_path() / "extension"
    
    graphs_dir = base_output_dir / "graphs"
    images_dir = base_output_dir / "images"
    
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"Processing: {image_path}")
    print(f"Output directory: {os.path.abspath(base_output_dir)}")

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
    print(f"Center found at: ({center_x}, {center_y})")

    # --- 3. Calculate Radial Profile ---
    radial_profile = ip.calculate_radial_profile(img_inverted, center_x, center_y)

    # --- 4. Coordinate Conversion ---
    radii_mm, pixel_scale_mm = calc.pixels_to_mm(radial_profile, phys_w_mm, phys_h_mm, (h, w))
    two_theta_deg = calc.mm_to_2theta(radii_mm, distance_L_mm)

    # --- 5. Normalization ---
    profile_percent = calc.normalize_profile(radial_profile, pixel_scale_mm)
    # User request: "don't smooth the intensity for the graphs"
    # We skip smoothing. This affects both plotting and peak finding.
    profile_smoothed = profile_percent 
    # profile_smoothed = calc.smooth_profile(profile_percent)

    # --- 6. Peak Finding ---
    # Use ring_radii as known peaks if available
    peak_indices, peak_properties, start_idx = calc.find_initial_peaks(profile_smoothed, known_peak_indices=ring_radii)
    
    # Estimate background by bridging
    # We pass the smoothed profile and the peak indices (adjusted for the start_idx offset if needed)
    background_profile = calc.estimate_background_bridging(two_theta_deg, profile_smoothed, peak_indices, peak_properties, start_idx)
    
    # Subtract background for fitting
    profile_subtracted = profile_smoothed - background_profile
    
    # --- 7. Sinc Fitting ---
    fitted_peaks = calc.fit_sinc_peaks(two_theta_deg, profile_subtracted, peak_indices, peak_properties)
    
    successful_fits = sum(1 for p in fitted_peaks if p is not None)
    print(f"Detected {len(peak_indices)} peaks. Successfully fitted {successful_fits} peaks.")

    # --- 8. Calculate d-spacings ---
    # Define peak_angles and peak_intensities here
    peak_angles = two_theta_deg[peak_indices]
    peak_intensities = profile_smoothed[peak_indices]
    d_spacings_pm = calc.calculate_d_spacings(peak_angles, wavelength_pm)
    
    # Calculate peak radii in pixels for visualization
    # peak_indices are indices into the radial_profile array, which corresponds to pixel radius
    peak_radii_pixels = peak_indices
    print(f"Peak radii (pixels): {peak_radii_pixels}")
    if len(peak_radii_pixels) > 0:
        print(f"  Min radius: {np.min(peak_radii_pixels)}")
        print(f"  Max radius: {np.max(peak_radii_pixels)}")
        print(f"  Image diagonal: {np.sqrt(h**2 + w**2)}")

    # --- 9. Plotting ---
    # Save center plot to 'images' directory
    # 1. With Pixels (Comparison)
    save_path_center_pixels = viz.plot_center_on_image(
        img_arr, center_x, center_y, peak_radii_pixels, base_name, str(images_dir), 
        centers_dict=centers_dict, filename_suffix="_center_pixels.png"
    )
    
    # 2. Clean (No Pixels)
    # Create a copy of centers_dict without the mask
    centers_dict_clean = centers_dict.copy()
    if 'ContourMask' in centers_dict_clean:
        del centers_dict_clean['ContourMask']
        
    save_path_center_clean = viz.plot_center_on_image(
        img_arr, center_x, center_y, peak_radii_pixels, base_name, str(images_dir), 
        centers_dict=centers_dict_clean, filename_suffix="_center_clean.png"
    )
    
    # Keep the original _center.png as the default (pixels) for backward compatibility/HTML
    save_path_center = save_path_center_pixels
    
    # 3. Emboldened (Visualization)
    # User request: "expect the opposite of blurring to make it sharper"
    save_path_emboldened = viz.save_emboldened_image(img_inverted, base_name, str(images_dir))
    
    # 4. Rings (Visualization)
    # User request: "peaks are converted back to rings... plotted on top of the image"
    save_path_rings = viz.plot_rings_on_image(img_inverted, (center_x, center_y), peak_indices, base_name, str(images_dir))
    
    # Save graphs to 'graphs' directory
    save_path_r = viz.plot_intensity_vs_radius(radii_mm, profile_percent, base_name, str(graphs_dir))
    save_path_theta = viz.plot_intensity_vs_2theta(
        two_theta_deg, profile_percent, profile_smoothed,
        peak_angles, peak_intensities, d_spacings_pm,
        fitted_peaks, background_profile,
        base_name, distance_L_mm, str(graphs_dir)
    )

    # --- 10. Save Data to CSV ---
    df = pd.DataFrame({
        'Radius_mm': radii_mm,
        'TwoTheta_deg': two_theta_deg,
        'Intensity_Raw': radial_profile,
        'Intensity_Normalized_Percent': profile_percent,
        'Intensity_Smoothed': profile_smoothed
    })

    # Save CSV to base output directory (or graphs? usually data is separate, but let's keep it in base or graphs)
    # User asked for graphs and images. Let's put CSV in base extension folder for now, or graphs?
    # "Keep them in a seperate directory - output/extenstion/graphs and output/extension/images"
    # It implies images go to images, graphs to graphs. CSV is data. I'll keep it in base extension folder to avoid cluttering graphs.
    csv_path = base_output_dir / f"{base_name}_analysis_data.csv"
    df.to_csv(csv_path, index=False)

    # --- 11. Generate Interactive Plots & HTML Report ---
    interactive_theta_fig = viz.create_interactive_plot(
        two_theta_deg, profile_percent, profile_smoothed,
        peak_angles, peak_intensities, d_spacings_pm,
        fitted_peaks, background_profile, base_name
    )
    
    interactive_radius_fig = viz.create_interactive_radius_plot(
        radii_mm, profile_percent, base_name
    )
    
    save_path_preprocessed = images_dir / f"{base_name}_preprocessed.png"
    
    html_path = viz.generate_html_report(
        str(base_output_dir), base_name, save_path_r, save_path_theta, str(csv_path), 
        interactive_theta_fig, interactive_radius_fig, save_path_center,
        preprocessed_image_path=str(save_path_preprocessed)
    )

    print(f"Saved plots and data to '{base_output_dir}' folder.")
    print(f"1. {Path(save_path_center).as_uri()}")
    print(f"2. {Path(save_path_r).as_uri()}")
    print(f"3. {Path(save_path_theta).as_uri()}")
    print(f"4. {Path(csv_path).as_uri()}")
    print(f"5. {Path(html_path).as_uri()}")

    return {
        'base_name': base_name,
        'plot_r_path': save_path_r,
        'plot_theta_path': save_path_theta,
        'plot_center_path': save_path_center, # Default (pixels)
        'plot_center_pixels_path': save_path_center_pixels,
        'plot_center_clean_path': save_path_center_clean,
        'plot_emboldened_path': save_path_emboldened,
        'plot_rings_path': save_path_rings,
        'csv_path': str(csv_path),
        'interactive_theta_fig': interactive_theta_fig,
        'interactive_radius_fig': interactive_radius_fig
    }
