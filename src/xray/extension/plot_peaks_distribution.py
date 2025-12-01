import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Ensure we can import from xray
project_root = Path(r"c:\Users\ofeke\Desktop\study\degree\year3\a\lab\a\XRay")
artifacts_dir = project_root / "artifacts" / "extension"
src_path = project_root / "src"
sys.path.append(str(src_path))

from xray.extension import calculations as calc

def main():
    files_map = [
        ("ChCl", "Solid_analysis_data.csv"),
        ("1 ChCL - 3 Urea", "7_20_analysis_data.csv"),
        ("1 ChCL - 10 Urea", "10_1_analysis_data.csv"), 
        ("Eutectic", "eutectic_analysis_data.csv")
    ]
    
    # Define colors for each sample
    colors = ['blue', 'green', 'purple', 'orange']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # We will plot on y-axis: 0, 1, 2, 3
    y_positions = range(len(files_map))
    labels = [f[0] for f in files_map]
    
    for i, (label, filename) in enumerate(files_map):
        file_path = artifacts_dir / filename
        color = colors[i % len(colors)]
        
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
            
        print(f"Processing {label} from {filename}...")
        df = pd.read_csv(file_path)
        
        two_theta = df['TwoTheta_deg'].values
        intensity_smoothed = df['Intensity_Smoothed'].values
        
        # Find peaks
        end_search_idx = np.searchsorted(two_theta, 45.0)
        if end_search_idx >= len(two_theta):
            end_search_idx = None
            
        # calc.find_initial_peaks now handles the log transformation internally.
        # We just pass the linear signal (amplified if needed).
        if label in ["ChCl", "Eutectic", "1 ChCL - 10 Urea"]:
             signal_for_peaks = intensity_smoothed * 20
        else:
             signal_for_peaks = intensity_smoothed
            
        if label in ["Eutectic", "1 ChCL - 10 Urea"]:
            # User request: "in eutenctic improve peak finding it is not, make for it the peak detenction find more peaks in area 0 to 10"
            # Applying same logic to 10_1
            idx_10 = np.searchsorted(two_theta, 10.0)
            
            p1, _, _ = calc.find_initial_peaks(
                signal_for_peaks,
                start_search_idx=0,
                end_search_idx=idx_10,
                prominence=0.01
            )
            
            p2, _, _ = calc.find_initial_peaks(
                signal_for_peaks,
                start_search_idx=idx_10,
                end_search_idx=end_search_idx,
                prominence=0.05
            )
            
            peak_indices = np.concatenate([p1, p2])
            peak_indices = np.sort(np.unique(peak_indices))
        else:
            peak_indices, peak_props, _ = calc.find_initial_peaks(
                signal_for_peaks,
                start_search_idx=0,
                end_search_idx=end_search_idx
            )
        
        # Calculate fit error (estimated from noise)
        # We estimate noise as std(Intensity_Raw - Intensity_Smoothed)
        # But we only have Intensity_Smoothed in CSV.
        # We need Intensity_Raw.
        if 'Intensity_Raw' in df.columns:
            intensity_raw = df['Intensity_Raw'].values
            residuals = intensity_raw - intensity_smoothed
            # Estimate noise from the lower 50% of residuals to avoid peaks
            # Or just take std of residuals
            fit_error = np.std(residuals)
        else:
            # Fallback if Raw not available (though it should be)
            fit_error = 0.5 
            
        # Calculate sigma_r
        # User formula: sqrt(2 + fit_error)
        # Assuming 2 is in same units as fit_error (intensity?) or maybe pixels?
        # If 2 is pixels, we need pixel_scale_mm.
        # Let's assume the user means "2 pixels variance + fit error variance" or similar.
        # But without pixel scale, we can't convert 2 pixels to mm.
        # However, we can estimate pixel_scale_mm from the radius array.
        # radii_mm is monotonic.
        
        radii_mm = df['Radius_mm'].values
        if len(radii_mm) > 1:
            pixel_scale_mm = np.mean(np.diff(radii_mm))
        else:
            pixel_scale_mm = 0.05 # Default guess
            
        # If "2" means "2 pixels", then in mm it is 2 * pixel_scale_mm?
        # Or maybe "2" is already in mm^2?
        # Let's assume the user means: sigma_r_mm = sqrt( (2*pixel_scale)**2 + (fit_error_mm)**2 )?
        # Or maybe simply: sigma_r = sqrt(2 + fit_error) is a value in some unit.
        # Given the ambiguity, and "get fit error from background fit",
        # I will use a robust estimation:
        # sigma_r_mm = pixel_scale_mm * np.sqrt(2 + fit_error) 
        # (Assuming fit_error is dimensionless or pixel-like).
        
        sigma_r_mm = pixel_scale_mm * np.sqrt(2 + fit_error)

        # Calculate L (distance)
        # L = r / tan(2theta)
        # Use a point with reasonable angle (e.g. max angle) to avoid division by zero near 0
        idx_L = -1
        if two_theta[idx_L] > 0.1:
            L_mm = radii_mm[idx_L] / np.tan(np.radians(two_theta[idx_L]))
        else:
            L_mm = 100.0 # Fallback
            
        # Calculate sigma_theta
        # User formula: arctan(r std / L)
        # sigma_theta_rad = arctan(sigma_r / L)
        sigma_theta_rad = np.arctan(sigma_r_mm / L_mm)
        sigma_theta_deg = np.degrees(sigma_theta_rad)
        
        peak_angles = two_theta[peak_indices]
        
        # Plot vertical lines for each peak at the specific y position
        y_center = i
        half_height = 0.3
        
        # User request: "make each line thicker"
        # User request: "add error bar in x"
        
        ax.errorbar(peak_angles, [y_center] * len(peak_angles), 
                    xerr=sigma_theta_deg, 
                    fmt='none', 
                    ecolor=color, 
                    elinewidth=2, # Thicker error bar
                    capsize=5)
                    
        # Plot the main line (marker) thicker
        ax.vlines(peak_angles, y_center - half_height, y_center + half_height, 
                  colors=color, linewidth=4) # Thicker line (was 2)
        
        # Optional: Add a horizontal line for the "row"
        ax.hlines(y_center, 0, 45, colors='gray', linestyles=':', alpha=0.3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("2-Theta (degrees)")
    ax.set_title("Peak Locations Distribution")
    ax.set_xlim(0, 45)
    ax.set_ylim(-0.5, len(files_map) - 0.5)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = artifacts_dir / "peaks_distribution.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved peaks distribution plot to {output_path}")

if __name__ == "__main__":
    main()
