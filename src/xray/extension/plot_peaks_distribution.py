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
    
    all_peaks_data = []
    
    print("Collecting peaks for distribution plot...")
    
    for i, (label, filename) in enumerate(files_map):
        file_path = artifacts_dir / filename
        
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
            
        if label in ["ChCl", "Eutectic", "1 ChCL - 10 Urea"]:
             signal_for_peaks = intensity_smoothed * 20
        else:
             signal_for_peaks = intensity_smoothed
            
        if label in ["Eutectic", "1 ChCL - 10 Urea"]:
            idx_10 = np.searchsorted(two_theta, 10.0)
            p1, _, _ = calc.find_initial_peaks(
                signal_for_peaks, start_search_idx=0, end_search_idx=idx_10, prominence=0.01
            )
            p2, _, _ = calc.find_initial_peaks(
                signal_for_peaks, start_search_idx=idx_10, end_search_idx=end_search_idx, prominence=0.05
            )
            peak_indices = np.concatenate([p1, p2])
            peak_indices = np.sort(np.unique(peak_indices))
        else:
            peak_indices, _, _ = calc.find_initial_peaks(
                signal_for_peaks, start_search_idx=0, end_search_idx=end_search_idx
            )
        
        # Calculate fit error
        if 'Intensity_Raw' in df.columns:
            intensity_raw = df['Intensity_Raw'].values
            residuals = intensity_raw - intensity_smoothed
            fit_error = np.std(residuals)
        else:
            fit_error = 0.5 
            
        radii_mm = df['Radius_mm'].values
        if len(radii_mm) > 1:
            pixel_scale_mm = np.mean(np.diff(radii_mm))
        else:
            pixel_scale_mm = 0.05
            
        sigma_r_mm = pixel_scale_mm * np.sqrt(2 + fit_error)

        # Ignore the first peak
        if len(peak_indices) > 1:
            peak_indices = peak_indices[1:]
        else:
            peak_indices = np.array([], dtype=int)

        # Calculate Widths
        current_widths_deg = []
        if len(peak_indices) > 0:
            widths_samples, _, left_ips, right_ips = calc.peak_widths(
                signal_for_peaks, peak_indices, rel_height=0.5
            )
            left_deg = np.interp(left_ips, np.arange(len(two_theta)), two_theta)
            right_deg = np.interp(right_ips, np.arange(len(two_theta)), two_theta)
            current_widths_deg = right_deg - left_deg
        
        # Get intensities
        peak_intensities = intensity_smoothed[peak_indices]
        
        # Normalize intensities by max(peaks[1:])
        if len(peak_intensities) > 0:
            max_intensity = np.max(peak_intensities)
            if max_intensity > 0:
                normalized_intensities = peak_intensities / max_intensity
            else:
                normalized_intensities = np.zeros_like(peak_intensities)
        else:
            normalized_intensities = np.array([])
            
        # Store peak data
        for k, idx in enumerate(peak_indices):
            theta = two_theta[idx]
            
            # Calculate L
            if theta > 0.1:
                L_mm = radii_mm[idx] / np.tan(np.radians(theta))
            else:
                L_mm = 100.0
                
            sigma_theta_rad = np.arctan(sigma_r_mm / L_mm)
            sigma_theta_deg = np.degrees(sigma_theta_rad)
            
            width_deg = current_widths_deg[k] if k < len(current_widths_deg) else 0.1
            
            all_peaks_data.append({
                'sample_idx': i,
                'theta': theta,
                'sigma': sigma_theta_deg,
                'intensity_norm': normalized_intensities[k],
                'width_deg': width_deg,
                'color': colors[i % len(colors)]
            })

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    y_positions = range(len(files_map))
    labels = [f[0] for f in files_map]
    
    # Calculate global max width for normalization
    all_widths = [p['width_deg'] for p in all_peaks_data]
    max_width = max(all_widths) if all_widths else 1.0
    if max_width == 0: max_width = 1.0
    
    # Draw rows
    for i in y_positions:
        ax.hlines(i, 0, 45, colors='gray', linestyles=':', alpha=0.3)
        
    for p in all_peaks_data:
        y = p['sample_idx']
        theta = p['theta']
        sigma = p['sigma']
        intensity_norm = p['intensity_norm']
        width_deg = p['width_deg']
        color = p['color']
        
        # Normalize width
        lw = (width_deg / max_width) * 6
        lw = np.clip(lw, 1.5, 8)
        
        max_half_height = 0.4
        half_heights = intensity_norm * max_half_height
        
        ax.errorbar(theta, y, xerr=sigma, fmt='none', ecolor=color, elinewidth=2, capsize=5)
        ax.vlines(theta, y - half_heights, y + half_heights, colors=color, linewidth=lw)

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
