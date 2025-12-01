import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Ensure we can import from xray
# Since this script is in src/xray/extension, we should be able to import directly if running as module,
# or add src to path if running as script.
current_dir = Path(__file__).parent # src/xray/extension
src_path = current_dir.parent.parent.parent # src
sys.path.append(str(src_path))

from xray.extension import calculations as calc

def main():
    # Define files and labels
    # User asked for: "eutenctic, 1_10, and 7_20 and solid"
    # We map them to the actual filenames found in artifacts/extension
    # Adjust this path to match the user's environment
    project_root = Path(r"c:\Users\ofeke\Desktop\study\degree\year3\a\lab\a\XRay")
    artifacts_dir = project_root / "artifacts" / "extension"
    
    files_map = [
        ("ChCl", "Solid_analysis_data.csv"),
        ("1 ChCL - 3 Urea", "7_20_analysis_data.csv"),
        ("1 ChCL - 10 Urea", "10_1_analysis_data.csv"), 
        ("Eutectic", "eutectic_analysis_data.csv")
    ]
    
    # Single plot for all
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for each sample
    colors = ['blue', 'green', 'purple', 'orange']
    
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
        
        # User request: "replace the red line with the ye,llow line"
        # In viz.py, the "yellow" (orange) line was smoothed * 20.
        intensity_to_plot = intensity_smoothed * 20
        
        # Find peaks using the new logic (45 degree limit)
        end_search_idx = np.searchsorted(two_theta, 45.0)
        if end_search_idx >= len(two_theta):
            end_search_idx = None
            
        # User request: "try to find the peaks in logaritmic scale"
        # calc.find_initial_peaks now handles the log transformation internally.
        if label in ["ChCl", "Eutectic", "1 ChCL - 10 Urea"]:
             signal_for_peaks = intensity_to_plot # This is smoothed * 20
        else:
             signal_for_peaks = intensity_smoothed
            
        if label in ["Eutectic", "1 ChCL - 10 Urea"]:
            # User request: "in eutenctic improve peak finding it is not, make for it the peak detenction find more peaks in area 0 to 10"
            # User request: "plus for some reason you find much less picks now in 10_1 care to look at that?"
            # Applying same logic to 10_1
            idx_10 = np.searchsorted(two_theta, 10.0)
            
            # Pass 1: 0-10 degrees, high sensitivity (prominence=0.01)
            p1, _, _ = calc.find_initial_peaks(
                signal_for_peaks,
                start_search_idx=0,
                end_search_idx=idx_10,
                prominence=0.01
            )
            
            # Pass 2: >10 degrees, normal sensitivity (prominence=0.05)
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
        
        # Plot Line
        ax.plot(two_theta, intensity_to_plot, color=color, linewidth=1.5, label=label)
        
        # Plot Peaks
        # User request: "match the colors of the peaks with the colors of the lines"
        peak_angles = two_theta[peak_indices]
        peak_intensities = intensity_to_plot[peak_indices]
        
        ax.plot(peak_angles, peak_intensities, "x", color=color, markersize=8, markeredgewidth=2)
        
        # Add labels for peaks (optional, might be too crowded)
        # for angle, intensity in zip(peak_angles, peak_intensities):
        #      ax.text(angle, intensity + 2, f"{angle:.1f}°", ha='center', va='bottom', fontsize=8, rotation=90, color=color)

    ax.set_xlabel("2-Theta (degrees)")
    ax.set_ylabel("Intensity (Scaled)")
    ax.set_title("Comparison of Diffraction Patterns")
    ax.grid(True, alpha=0.3)
    # ax.set_ylim(0, 110) # Removed fixed limit as *20 might exceed 100
    ax.set_xlim(0, 45)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    output_path = artifacts_dir / "comparison_peaks_combined.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved comparison plot to {output_path}")

if __name__ == "__main__":
    main()
