import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

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
    
    colors = ['blue', 'green', 'purple', 'orange']
    
    avg_intensities = []
    avg_widths = []
    labels = []
    
    print("Calculating peak statistics...")
    
    for i, (label, filename) in enumerate(files_map):
        file_path = artifacts_dir / filename
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        two_theta = df['TwoTheta_deg'].values
        intensity_smoothed = df['Intensity_Smoothed'].values
        
        # Peak Finding Logic
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
        
        # Ignore first peak
        if len(peak_indices) > 1:
            peak_indices = peak_indices[1:]
        else:
            peak_indices = np.array([], dtype=int)
            
        # Calculate Stats
        if len(peak_indices) > 0:
            # Intensity
            # Use original smoothed intensity, not the amplified one used for peak finding
            current_peak_intensities = intensity_smoothed[peak_indices]
            avg_int = np.mean(current_peak_intensities)
            
            # Width
            widths_samples, _, left_ips, right_ips = calc.peak_widths(
                signal_for_peaks, peak_indices, rel_height=0.5
            )
            left_deg = np.interp(left_ips, np.arange(len(two_theta)), two_theta)
            right_deg = np.interp(right_ips, np.arange(len(two_theta)), two_theta)
            current_widths_deg = right_deg - left_deg
            avg_wid = np.mean(current_widths_deg)
        else:
            avg_int = 0.0
            avg_wid = 0.0
            
        avg_intensities.append(avg_int)
        avg_widths.append(avg_wid)
        labels.append(label)
        
    # Plotting
    # 1. Average Intensity
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(labels, avg_intensities, color=colors)
    ax1.set_ylabel("Average Intensity")
    ax1.set_title("Average Peak Intensity per Sample")
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    output_int = artifacts_dir / "avg_peak_intensity.png"
    plt.savefig(output_int, dpi=300)
    print(f"Saved {output_int}")
    
    # 2. Average Width
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(labels, avg_widths, color=colors)
    ax2.set_ylabel("Average Width (degrees)")
    ax2.set_title("Average Peak Width per Sample")
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    output_wid = artifacts_dir / "avg_peak_width.png"
    plt.savefig(output_wid, dpi=300)
    print(f"Saved {output_wid}")

if __name__ == "__main__":
    main()
