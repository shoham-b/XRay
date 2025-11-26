import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

def pixels_to_mm(radial_profile, phys_w_mm, phys_h_mm, img_shape):
    """Converts pixel coordinates to mm."""
    h, w = img_shape
    scale_x = phys_w_mm / w
    scale_y = phys_h_mm / h
    pixel_scale_mm = (scale_x + scale_y) / 2

    radii_pixels = np.arange(len(radial_profile))
    radii_mm = radii_pixels * pixel_scale_mm
    return radii_mm, pixel_scale_mm

def mm_to_2theta(radii_mm, distance_L_mm):
    """Converts mm radius to 2-theta in degrees."""
    two_theta_rad = np.arctan(radii_mm / distance_L_mm)
    two_theta_deg = np.degrees(two_theta_rad)
    return two_theta_deg

def normalize_profile(radial_profile, pixel_scale_mm):
    """Subtracts a simple minimum background and normalizes to 100%."""
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
    
    return profile_percent

def smooth_profile(profile_percent, window=15):
    """Smooths the profile using Savitzky-Golay filter."""
    if len(profile_percent) > window:
        return savgol_filter(profile_percent, window, 3)
    return profile_percent

def find_initial_peaks(profile_smoothed, prominence=0.5, distance=10, known_peak_indices=None):
    """
    Finds peaks in the smoothed profile.
    If known_peak_indices are provided (e.g. from ring fitting), uses them as a base.
    """
    if known_peak_indices is not None and len(known_peak_indices) > 0:
        # Refine the known indices by looking for local maxima near them
        refined_indices = []
        search_window = 10
        for idx in known_peak_indices:
            idx = int(idx)
            start = max(0, idx - search_window)
            end = min(len(profile_smoothed), idx + search_window)
            # Find max in this window
            local_max_offset = np.argmax(profile_smoothed[start:end])
            refined_indices.append(start + local_max_offset)
        
        peak_indices = np.array(refined_indices)
        # We still need properties for width calculation
        # We can run find_peaks just to get properties for these specific peaks?
        # Or just run find_peaks globally and match?
        # Let's run find_peaks globally with loose parameters to get properties, 
        # then filter for our known peaks.
        all_peaks, all_props = find_peaks(profile_smoothed, prominence=0.1, width=1)
        
        # Match refined_indices to all_peaks to get properties
        final_indices = []
        final_props = {k: [] for k in all_props}
        
        for ri in peak_indices:
            # Find closest peak in all_peaks
            closest_idx = np.argmin(np.abs(all_peaks - ri))
            if np.abs(all_peaks[closest_idx] - ri) < 5:
                final_indices.append(all_peaks[closest_idx])
                for k in all_props:
                    final_props[k].append(all_props[k][closest_idx])
            else:
                # If no peak found by find_peaks near our known peak, just use the known one
                # and estimate width manually or skip?
                # Let's add it but with default width
                final_indices.append(ri)
                final_props['widths'].append(5.0) # Default width
                final_props['left_ips'].append(max(0, ri-2))
                final_props['right_ips'].append(min(len(profile_smoothed), ri+2))
                # Add other keys if needed, but width is most important
        
        peak_indices = np.array(final_indices)
        peak_properties = {k: np.array(v) for k, v in final_props.items()}
        
    else:
        # Standard peak finding
        peak_indices, peak_properties = find_peaks(profile_smoothed, prominence=prominence, distance=distance, width=1)
    
    # Filter out peaks near the start (beam stop)
    start_idx = 20
    valid_mask = peak_indices > start_idx
    peak_indices = peak_indices[valid_mask]
    for key in peak_properties:
        peak_properties[key] = peak_properties[key][valid_mask]

    return peak_indices, peak_properties, start_idx

def estimate_background_bridging(x_data, y_data, peak_indices, peak_properties, start_idx):
    """
    Estimates background by bridging over peaks.
    Uses peak widths to identify the bases.
    """
    y_bg = y_data.copy()
    
    # Sort peaks
    sorted_indices = np.argsort(peak_indices)
    sorted_peaks = peak_indices[sorted_indices]
    
    # Extract widths and bases from properties (relative to start_idx)
    # We need to map them back to absolute indices
    # properties['left_ips'] and 'right_ips' give interpolated positions
    # Let's use 'widths' to define a window
    
    widths = peak_properties['widths'][sorted_indices]
    
    for i, idx in enumerate(sorted_peaks):
        w = widths[i]
        # Define a window based on peak width (e.g., 3x width) to find true background
        half_window = int(w * 2.0) 
        
        left = max(0, idx - half_window)
        right = min(len(x_data) - 1, idx + half_window)
        
        if left >= right:
            continue
            
        # Find the minimum value in the left and right regions to anchor the bridge
        # This avoids anchoring on the slope of the peak
        
        # Left anchor: min in [left, idx - w/2]
        left_region_end = max(left + 1, int(idx - w/2))
        if left < left_region_end:
            left_anchor_idx = left + np.argmin(y_data[left:left_region_end])
        else:
            left_anchor_idx = left
            
        # Right anchor: min in [idx + w/2, right]
        right_region_start = min(right - 1, int(idx + w/2))
        if right_region_start < right:
            right_anchor_idx = right_region_start + np.argmin(y_data[right_region_start:right])
        else:
            right_anchor_idx = right
            
        x1, y1 = x_data[left_anchor_idx], y_data[left_anchor_idx]
        x2, y2 = x_data[right_anchor_idx], y_data[right_anchor_idx]
        
        if x2 > x1:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            bridge_indices = np.arange(left_anchor_idx + 1, right_anchor_idx)
            if bridge_indices.size > 0:
                y_bg[bridge_indices] = m * x_data[bridge_indices] + b
                
    # Smooth the background
    y_bg = savgol_filter(y_bg, window_length=51, polyorder=3)
    y_bg = np.minimum(y_bg, y_data)
    
    return y_bg

def sinc_func(x, amp, center, sigma, offset):
    """
    Sinc function: A * sinc((x - x0) / sigma) + offset
    """
    return amp * np.sinc((x - center) / sigma) + offset

def fit_sinc_peaks(x_data, y_data_subtracted, peak_indices, peak_properties):
    """
    Fits Sinc functions to the background-subtracted peaks.
    """
    fitted_peaks = []
    widths = peak_properties['widths']
    
    for i, idx in enumerate(peak_indices):
        center_guess = x_data[idx]
        amp_guess = y_data_subtracted[idx]
        
        # Estimate sigma from peak width (FWHM approx)
        if idx < len(x_data) - 1:
            step = x_data[idx+1] - x_data[idx]
        else:
            step = x_data[idx] - x_data[idx-1]
            
        width_deg = widths[i] * step
        sigma_guess = max(width_deg / 1.2, 0.05) # Ensure sigma isn't too small
        
        # Define window for fitting (e.g., 4x sigma)
        window_deg = max(width_deg * 4, 2.0)
        
        mask = (x_data >= center_guess - window_deg/2) & (x_data <= center_guess + window_deg/2)
        x_window = x_data[mask]
        y_window = y_data_subtracted[mask]
        
        if len(x_window) < 5:
            print(f"Skipping peak at {center_guess:.2f}: too few points ({len(x_window)})")
            fitted_peaks.append(None)
            continue
            
        try:
            # Relaxed bounds for offset and sigma
            # Offset can be up to 50% of amplitude (positive or negative)
            popt, _ = curve_fit(sinc_func, x_window, y_window, 
                                p0=[amp_guess, center_guess, sigma_guess, 0],
                                bounds=([0, center_guess - window_deg/2, 0.01, -amp_guess*0.5], 
                                        [np.inf, center_guess + window_deg/2, window_deg*2, amp_guess*0.5]),
                                maxfev=10000)
            fitted_peaks.append(popt)
            # print(f"Fit success for peak at {center_guess:.2f}")
        except Exception as e:
            print(f"Fit failed for peak at {center_guess:.2f}: {e}")
            fitted_peaks.append(None)
            
    return fitted_peaks

def calculate_d_spacings(two_theta_deg, wavelength_pm):
    """Calculates d-spacings using Bragg's Law."""
    theta_rad = np.radians(two_theta_deg / 2.0)
    # Avoid division by zero
    with np.errstate(divide='ignore'):
        d_spacings_pm = wavelength_pm / (2 * np.sin(theta_rad))
    return d_spacings_pm
