import pandas as pd
import numpy as np
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
    
    for i, (label, filename) in enumerate(files_map):
        file_path = artifacts_dir / filename
        if not file_path.exists():
            continue
            
        df = pd.read_csv(file_path)
        two_theta = df['TwoTheta_deg'].values
        intensity_smoothed = df['Intensity_Smoothed'].values
        
        # Peak Finding Logic
        end_search_idx = np.searchsorted(two_theta, 45.0)
        if end_search_idx >= len(two_theta): end_search_idx = None
            
        if label in ["ChCl", "Eutectic", "1 ChCL - 10 Urea"]:
            signal_for_peaks = intensity_smoothed * 20
        else:
            signal_for_peaks = intensity_smoothed
            
        if label in ["Eutectic", "1 ChCL - 10 Urea"]:
            idx_10 = np.searchsorted(two_theta, 10.0)
            p1, _, _ = calc.find_initial_peaks(signal_for_peaks, start_search_idx=0, end_search_idx=idx_10, prominence=0.01)
            p2, _, _ = calc.find_initial_peaks(signal_for_peaks, start_search_idx=idx_10, end_search_idx=end_search_idx, prominence=0.05)
            peak_indices = np.concatenate([p1, p2])
            peak_indices = np.sort(np.unique(peak_indices))
        else:
            peak_indices, _, _ = calc.find_initial_peaks(signal_for_peaks, start_search_idx=0, end_search_idx=end_search_idx)
        
        # Ignore first peak
        if len(peak_indices) > 1:
            peak_indices = peak_indices[1:]
        else:
            peak_indices = np.array([], dtype=int)
            
        current_peak_intensities = signal_for_peaks[peak_indices]
        if len(current_peak_intensities) > 0:
            max_val = np.max(current_peak_intensities)
            norm_intensities = current_peak_intensities / max_val
            
            print(f"\nSample: {label}")
            print(f"Max Intensity: {max_val}")
            print("Normalized Intensities:")
            for j, val in enumerate(norm_intensities):
                print(f"  Peak {j}: {val:.4f} (Angle: {two_theta[peak_indices[j]]:.2f})")
        else:
            print(f"\nSample: {label} - No peaks after ignoring first.")

if __name__ == "__main__":
    main()
