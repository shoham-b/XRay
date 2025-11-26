import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter, find_peaks

def remove_blue(image_path):
    # 1. Open image and ensure it is RGB (3 channels) or RGBA (4 channels)
    raw = Image.open(image_path).convert("RGB")
    
    # 2. Convert to NumPy array
    data = np.array(raw)
    
    # 3. Set the Blue channel to 0
    # Syntax: [all rows, all columns, channel index 2] (0=R, 1=G, 2=B)
    data[:, :, 2] = 0
    
    # 4. Convert back to Image and return
    return Image.fromarray(data)

def process_diffraction_image(image_path, phys_w_mm, phys_h_mm, distance_L_mm, wavelength_pm=71.1):
    """
    Analyzes X-ray film, generates plots, and saves results to a 'data' folder.
    """
    
    # --- 0. Setup Output Directory & Filenames ---
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"Processing: {image_path}")
    print(f"Output directory: {os.path.abspath(output_dir)}")

    # --- 1. Load and Preprocess Image ---
    try:
        raw = Image.open(image_path)
        img = remove_blue(raw).convert('L')
    except FileNotFoundError:
        print(f"Error: File {image_path} not found.")
        return

    img_arr = np.array(img)
    # Invert: Film is negative (dark = high intensity)
    img_inverted = 255.0 - img_arr 
    h, w = img_arr.shape
    
    # --- 2. Find Center (Highest Intensity Point) ---
    # Smooth slightly and mask edges to find the robust beam center
    img_smoothed = gaussian_filter(img_inverted, sigma=5)
    
    border = 50 # Mask 50 pixels from edges to avoid scanning artifacts
    mask = np.zeros_like(img_smoothed)
    mask[border:h-border, border:w-border] = 1
    masked_img = img_smoothed * mask
    
    cy, cx = np.unravel_index(np.argmax(masked_img), masked_img.shape)
    center_x, center_y = int(cx), int(cy)
    print(f"Center found at: ({center_x}, {center_y})")
    
    # --- 3. Calculate Radial Profile ---
    y, x = np.indices(img_arr.shape)
    r_pixels = np.sqrt((x - center_x)*2 + (y - center_y)*2)
    r_pixels_int = r_pixels.astype(int)
    
    # Azimuthal integration (Average intensity at radius r)
    tbin = np.bincount(r_pixels_int.ravel(), img_inverted.ravel())
    nr = np.bincount(r_pixels_int.ravel())
    nr[nr == 0] = 1
    radial_profile = tbin / nr
    
    # --- 4. Coordinate Conversion ---
    # Convert Pixels -> mm
    scale_x = phys_w_mm / w
    scale_y = phys_h_mm / h
    pixel_scale_mm = (scale_x + scale_y) / 2 
    
    radii_pixels = np.arange(len(radial_profile))
    radii_mm = radii_pixels * pixel_scale_mm
    
    # Convert mm -> 2-Theta (Degrees)
    # tan(2theta) = r / L
    two_theta_rad = np.arctan(radii_mm / distance_L_mm)
    theta_deg = np.degrees(two_theta_rad / 2)
    
    # --- 5. Normalization (0% to 100%) ---
    # Subtract background (estimated from outer region)
    valid_start_idx = int(100 / pixel_scale_mm)
    if valid_start_idx < len(radial_profile):
        min_bg = np.min(radial_profile[valid_start_idx:])
    else:
        min_bg = 0
        
    profile_corrected = radial_profile - min_bg
    profile_corrected[profile_corrected < 0] = 0
    
    # Normalize: Max Intensity (Primary Beam) = 100%
    max_val = np.max(profile_corrected)
    if max_val == 0: max_val = 1
    profile_percent = (profile_corrected / max_val) * 100.0
    
    # Smooth for cleaner plotting/peak finding
    window = 15
    if len(profile_percent) > window:
        profile_smoothed = savgol_filter(profile_percent, window, 3)
    else:
        profile_smoothed = profile_percent

    # --- 6. Plot 1: Intensity vs. Radius (mm) ---
    plt.figure(figsize=(10, 6))
    plt.plot(radii_mm, profile_percent, color='blue', linewidth=1.5, label='Avg Intensity')
    plt.title(f"Normalized Intensity vs. Distance ($r$)\nSample: {base_name}")
    plt.xlabel("Distance from Center $r$ (mm)")
    plt.ylabel("Normalized Intensity (%)")
    plt.xlim(0, np.max(radii_mm))
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.legend()
    
    save_path_r = os.path.join(output_dir, f"{base_name}_intensity_vs_radius.png")
    plt.savefig(save_path_r)
    plt.close() # Close to free memory
    
    # --- 7. Plot 2: Intensity vs. 2-Theta with d-spacings ---
    # Find Peaks (ignoring the primary beam < 3 degrees)
    min_angle = 3.0
    start_idx = np.argmax(theta_deg > min_angle)
    
    peaks, _ = find_peaks(profile_smoothed[start_idx:], prominence=0.5, distance=10)
    peak_indices = peaks + start_idx
    
    peak_angles = theta_deg[peak_indices]
    peak_intensities = profile_smoothed[peak_indices]
    
    # Calculate d-spacings (Bragg's Law)
    # theta is half of 2-theta
    theta_rad = np.radians(peak_angles / 2.0)
    d_spacings_pm = wavelength_pm / (2 * np.sin(theta_rad))
    
    plt.figure(figsize=(12, 7))
    plt.plot(theta_deg, profile_percent, color='lightgray', label='Raw Data')
    plt.plot(theta_deg, profile_smoothed, color='darkblue', linewidth=2, label='Smoothed Intensity')
    
    # Label Peaks
    for i, angle in enumerate(peak_angles):
        intensity = peak_intensities[i]
        d_val = d_spacings_pm[i]
        plt.plot(angle, intensity, "x", color='red', markersize=8)
        plt.text(angle, intensity + 2, f"d={d_val:.0f} pm\n({angle:.1f}°)", 
                 ha='center', va='bottom', fontsize=9, color='darkred', rotation=90)
                 
    plt.title(f"Diffraction Pattern: Intensity vs. $\\theta$ (with d-spacings)\nSample: {base_name} | L={distance_L_mm}mm")
    plt.xlabel("Scattering Angle $\\theta$ (degrees)")
    plt.ylabel("Relative Intensity (%)")
    # plt.xlim(0, 15) 
    # plt.ylim(0, 115)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    save_path_theta = os.path.join(output_dir, f"{base_name}_intensity_vs_theta.png")
    plt.savefig(save_path_theta)
    plt.close()

    # --- 8. Save Data to CSV ---
    # Create a DataFrame to hold the profile data
    df = pd.DataFrame({
        'Radius_mm': radii_mm,
        'TwoTheta_deg': theta_deg,
        'Intensity_Raw': radial_profile,
        'Intensity_Normalized_Percent': profile_percent
    })
    
    csv_path = os.path.join(output_dir/"data", f"{base_name}_analysis_data.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Saved plots and data to '{output_dir}' folder.")
    print(f"1. {os.path.basename(save_path_r)}")
    print(f"2. {os.path.basename(save_path_theta)}")
    print(f"3. {os.path.basename(csv_path)}")

# --- Execution ---
# Ensure your image file is in the same folder as this script
PHYS_WIDTH_MM = 55.0
PHYS_HEIGHT_MM = 75.0


FILENAME = "raw/7_20.jpg"  
DISTANCE_L_MM = 20.0
process_diffraction_image(FILENAME, PHYS_WIDTH_MM, PHYS_HEIGHT_MM, DISTANCE_L_MM)

FILENAME = "raw/Liquid.jpg"  
DISTANCE_L_MM = 10.0
process_diffraction_image(FILENAME, PHYS_WIDTH_MM, PHYS_HEIGHT_MM, DISTANCE_L_MM)

FILENAME = "raw/Solid.jpg"  
DISTANCE_L_MM = 15.0
process_diffraction_image(FILENAME, PHYS_WIDTH_MM, PHYS_HEIGHT_MM, DISTANCE_L_MM)

FILENAME = "raw/SALT.jpg"  
DISTANCE_L_MM = 15.0
process_diffraction_image(FILENAME, PHYS_WIDTH_MM, PHYS_HEIGHT_MM, DISTANCE_L_MM)

FILENAME = "raw/10_1.jpg"  
DISTANCE_L_MM = 20.0
process_diffraction_image(FILENAME, PHYS_WIDTH_MM, DISTANCE_L_MM, PHYS_HEIGHT_MM)