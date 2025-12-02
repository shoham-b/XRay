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
from scipy.signal import find_peaks

def main():
    files_map = [
        ("ChCl", "Solid_analysis_data.csv"),
        ("1 ChCL - 3 Urea", "7_20_analysis_data.csv"),
        ("1 ChCL - 10 Urea", "10_1_analysis_data.csv"), 
        ("Eutectic", "eutectic_analysis_data.csv")
    ]
    
    colors = ['blue', 'green', 'purple', 'orange']
    
    # 1. Collect all peaks
    all_peaks = [] # List of (theta, sample_index, intensity, sigma_theta)
    
    print("Collecting peaks...")
    for i, (label, filename) in enumerate(files_map):
        file_path = artifacts_dir / filename
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        two_theta = df['TwoTheta_deg'].values
        intensity_smoothed = df['Intensity_Smoothed'].values
        radii_mm = df['Radius_mm'].values
        
        # Peak Finding Logic (same as plot_peaks_distribution.py)
        end_search_idx = np.searchsorted(two_theta, 45.0)
        if end_search_idx >= len(two_theta):
            end_search_idx = None
            
        # Use amplified signal for ChCl and Eutectic and 10_1
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
        
        # Calculate sigma_theta (error)
        # fit_error logic
        if 'Intensity_Raw' in df.columns:
            intensity_raw = df['Intensity_Raw'].values
            residuals = intensity_raw - intensity_smoothed
            fit_error = np.std(residuals)
        else:
            fit_error = 0.5
            
        if len(radii_mm) > 1:
            pixel_scale_mm = np.mean(np.diff(radii_mm))
        else:
            pixel_scale_mm = 0.05
            
        sigma_r_mm = pixel_scale_mm * np.sqrt(2 + fit_error)
        
        # Ignore first peak
        if len(peak_indices) > 1:
            peak_indices = peak_indices[1:]
        else:
            peak_indices = np.array([], dtype=int)
            
        current_peak_intensities = signal_for_peaks[peak_indices]
        if len(current_peak_intensities) > 0:
            max_val = np.max(current_peak_intensities)
            if max_val > 0:
                norm_intensities = current_peak_intensities / max_val
            else:
                norm_intensities = np.zeros_like(current_peak_intensities)
        else:
            norm_intensities = []
        
        # Collect peaks
        for k, idx in enumerate(peak_indices):
            theta = two_theta[idx]
            intensity_norm = norm_intensities[k]
            
            # Calculate L for this peak
            if theta > 0.1:
                L_mm = radii_mm[idx] / np.tan(np.radians(theta))
            else:
                L_mm = 100.0
                
            sigma_theta_rad = np.arctan(sigma_r_mm / L_mm)
            sigma_theta_deg = np.degrees(sigma_theta_rad)
            
            all_peaks.append({
                'theta': theta,
                'sample_idx': i,
                'label': label,
                'color': colors[i],
                'sigma': sigma_theta_deg,
                'intensity_norm': intensity_norm
            })

    # 2. Group Peaks
    # User request: "make it so it choses only the best match (closest within error range)"
    # User request: "it cannot be that two picks from the same image are common with other two picks"
    # Implies constrained clustering: a group cannot contain >1 peak from same sample.
    
    all_peaks.sort(key=lambda x: x['theta'])
    
    # List of groups. Each group is a dict: {'center': theta, 'peaks': [p1, p2...], 'samples': set(sample_indices)}
    groups = []
    
    for p in all_peaks:
        # Try to find a compatible group
        best_group = None
        min_dist = float('inf')
        
        for g in groups:
            # Check constraint: sample not in group
            if p['sample_idx'] in g['samples']:
                continue
                
            # Check tolerance
            # Use max sigma of group + current peak as tolerance? Or fixed?
            # User said "within range of error".
            # Let's use max(p.sigma, g.sigma_max, 0.3)
            # We need to track sigma_max of group
            
            tolerance = max(p['sigma'], g['max_sigma'], 0.3)
            dist = abs(p['theta'] - g['center'])
            
            if dist <= tolerance:
                if dist < min_dist:
                    min_dist = dist
                    best_group = g
        
        if best_group:
            # Add to best group
            best_group['peaks'].append(p)
            best_group['samples'].add(p['sample_idx'])
            # Update center (running average)
            n = len(best_group['peaks'])
            best_group['center'] = (best_group['center'] * (n-1) + p['theta']) / n
            best_group['max_sigma'] = max(best_group['max_sigma'], p['sigma'])
        else:
            # Create new group
            groups.append({
                'center': p['theta'],
                'peaks': [p],
                'samples': {p['sample_idx']},
                'max_sigma': p['sigma']
            })
            
    # Convert groups back to list of lists for plotting logic
    final_groups = [g['peaks'] for g in groups]
        
    # 3. Classify Groups
    common_peaks = [] # Groups with >= 3 unique samples
    rare_peaks = []   # Groups with < 3 unique samples
    
    # User request: "change colors of the picks in such a way that each group of coomon peaks will have the same color"
    # We need to assign a color to each group.
    # We can use a colormap.
    cmap = plt.get_cmap('tab20') # 20 distinct colors
    
    for i, g in enumerate(final_groups):
        # Assign color to all peaks in this group
        group_color = cmap(i % 20)
        
        # Update color in peak dicts
        for p in g:
            p['group_color'] = group_color
            
        if len(g) >= 3:
            common_peaks.extend(g)
        else:
            rare_peaks.extend(g)
            
    # 4. Plotting
    def plot_peaks(peaks_list, title, filename):
        fig, ax = plt.subplots(figsize=(12, 6))
        y_positions = range(len(files_map))
        labels = [f[0] for f in files_map]
        
        # Draw rows
        for i in y_positions:
            ax.hlines(i, 0, 45, colors='gray', linestyles=':', alpha=0.3)
            
        for p in peaks_list:
            y = p['sample_idx']
            # Use group_color if available, else sample color (though all should have group_color now)
            color = p.get('group_color', p['color'])
            theta = p['theta']
            sigma = p['sigma']
            intensity_norm = p.get('intensity_norm', 1.0)
            
            # Ensure minimum visibility for weak peaks
            # User reported "intensities disappeared", likely due to small normalized values.
            # We enforce a minimum half_height of 0.1 (total height 0.2)
            min_half_height = 0.1
            half_height = max(intensity_norm * 0.4, min_half_height)
            
            ax.errorbar(theta, y, xerr=sigma, fmt='none', ecolor=color, elinewidth=2, capsize=5)
            ax.vlines(theta, y - half_height, y + half_height, colors=color, linewidth=4)
            
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel("2-Theta (degrees)")
        ax.set_title(title)
        ax.set_xlim(0, 45)
        ax.set_ylim(-0.5, len(files_map) - 0.5)
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = artifacts_dir / filename
        plt.savefig(output_path, dpi=300)
        print(f"Saved {title} to {output_path}")

    plot_peaks(common_peaks, "Common Peaks (in >= 3 samples)", "peaks_common.png")
    plot_peaks(rare_peaks, "Rare Peaks (in < 3 samples)", "peaks_rare.png")

if __name__ == "__main__":
    main()
