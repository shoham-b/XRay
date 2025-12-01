import numpy as np
import logging
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from scipy.ndimage import minimum_filter

logger = logging.getLogger(__name__)

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

def find_initial_peaks(profile_smoothed, prominence=0.05, distance=10, known_peak_indices=None, start_search_idx=20, end_search_idx=None):
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
        # User request: "before finding peaks add to the graph min(data) so no point will have negative intensity"
        # Ensure profile is positive before log
        min_val = np.min(profile_smoothed)
        if min_val <= 0:
            profile_shifted = profile_smoothed - min_val + 1.0
        else:
            profile_shifted = profile_smoothed
            
        # User request: "try to find the peaks in logaritmic scale"
        peak_indices, peak_properties = find_peaks(np.log2(profile_shifted), prominence=prominence, distance=distance, width=0.5)
    
    # Filter out peaks near the start (beam stop) and end
    # User request: "find peaks at the begginign now , and just ignore the first peak"
    
    # If start_search_idx is small (meaning we are searching from the beginning),
    # we should filter by index but also drop the very first peak found.
    
    valid_mask = peak_indices > start_search_idx
    if end_search_idx is not None:
        valid_mask &= peak_indices < end_search_idx
        
    peak_indices = peak_indices[valid_mask]
    for key in peak_properties:
        peak_properties[key] = peak_properties[key][valid_mask]

    # Drop the first peak if we have any and we are searching from the start
    # User request: "show all picks everytime, dont ignore first pick as you do today"
    # Logic removed.
    
    return peak_indices, peak_properties, start_search_idx

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
            logger.warning(f"Skipping peak at {center_guess:.2f}: too few points ({len(x_window)})")
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
            logger.warning(f"Fit failed for peak at {center_guess:.2f}: {e}")
            fitted_peaks.append(None)
            
    return fitted_peaks

def calculate_d_spacings(two_theta_deg, wavelength_pm):
    """Calculates d-spacings using Bragg's Law."""
    theta_rad = np.radians(two_theta_deg / 2.0)
    # Avoid division by zero
    theta_rad[theta_rad == 0] = 1e-9
    d_spacings = wavelength_pm / (2 * np.sin(theta_rad))
    return d_spacings

def fit_convex_hull_background(radii_mm, intensity, saturation_threshold=97, degree=6, sigma=3.0, max_iterations=20):
    """
    Fits a background to the intensity profile using iterative piecewise linear interpolation.
    Renamed from 'polynomial' to 'convex_hull' to reflect the actual algorithm.
    """
    max_intensity = np.max(intensity)
    threshold_val = (saturation_threshold / 100.0) * max_intensity
    
    # Find the start index for fitting (skip saturated region)
    # Logic: Find the first saturated point, then find the first valid point after that.
    # This handles cases like:
    # 1. High -> Low (Saturation at start)
    # 2. Low -> High -> Low (Beam stop then Saturation)
    
    is_saturated = intensity >= threshold_val
    
    if np.any(is_saturated):
        # Find start of first saturated region
        first_sat_idx = np.argmax(is_saturated)
        
        # Check for valid data AFTER this saturation
        is_valid_after = intensity[first_sat_idx:] < threshold_val
        
        if np.any(is_valid_after):
            # Found valid data after saturation
            offset = np.argmax(is_valid_after)
            # User request: "start the background fit, after a smal range of 10 below the required intensity"
            # We add a buffer of 10 indices to ensure we are off the steep slope
            start_idx = first_sat_idx + offset + 5
            
            # Ensure we don't go out of bounds
            if start_idx >= len(intensity):
                start_idx = len(intensity) - 1
        else:
            # Saturated until the end
            start_idx = len(intensity)
    else:
        # No saturation found
        start_idx = 0
        
    # Trim the first 1 mm of data
    if len(radii_mm) > 0:
        start_radius = radii_mm[0]
        trim_mask = radii_mm >= start_radius + 1.0
        if np.any(trim_mask):
            trim_idx = np.argmax(trim_mask)
            start_idx = max(start_idx, trim_idx)

    # Use data from start_idx onwards
    r_data = radii_mm[start_idx:]
    y_data = intensity[start_idx:]

    if len(r_data) < 2:
        return np.zeros_like(intensity), start_idx

    try:
        # --- Part 1: Initial Region (Upper Convex Hull of Lower Envelope) ---
        # User request: "Until then [start_idx], use an upside convex in the other direction"
        # User request: "the first upward convex should also be below the graph"
        bg_init_profile = np.zeros(start_idx)
        
        if start_idx > 2:
            r_init = radii_mm[:start_idx]
            y_init = intensity[:start_idx]
            
            # 1. Compute Lower Envelope (to ensure "below the graph")
            # Use a window size appropriate for the data density
            window_len_init = 11
            if window_len_init > len(y_init): window_len_init = len(y_init)
            
            y_init_lower = minimum_filter(y_init, size=window_len_init)
            
            # 2. Compute Upper Convex Hull of this Lower Envelope
            # Points sorted by x
            points_init = list(zip(r_init, y_init_lower))
            
            def cross_product(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

            upper_hull = []
            for p in points_init:
                # For upper hull, we want right turns. 
                # If we make a left turn (cross_product > 0), we pop.
                # (Assuming standard coordinates where y increases upwards)
                while len(upper_hull) >= 2 and cross_product(upper_hull[-2], upper_hull[-1], p) >= 0:
                    upper_hull.pop()
                upper_hull.append(p)
            
            # Interpolate Upper Hull
            if len(upper_hull) > 0:
                r_hull_init, y_hull_init = zip(*upper_hull)
                bg_init_profile = np.interp(r_init, r_hull_init, y_hull_init)


        # --- Part 2: Main Region (Lower Convex Hull) ---
        # 1. Smooth data strongly to define the "lower envelope" shape without noise
        window_len = 51
        if window_len > len(y_data):
            window_len = len(y_data)
        if window_len % 2 == 0:
            window_len -= 1
        if window_len < 5:
            window_len = 5
            
        y_smooth_fit = savgol_filter(y_data, window_length=window_len, polyorder=3)
        
        # 2. Compute Lower Convex Hull of (r_data, y_smooth_fit)
        # Points must be sorted by x (r_data is sorted)
        points = list(zip(r_data, y_smooth_fit))
        
        # (cross_product defined above)

        hull = []
        for p in points:
            # For lower hull, we want left turns (convex). 
            # If we make a right turn (or straight), we pop.
            # Right turn means cross_product <= 0
            while len(hull) >= 2 and cross_product(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
            
        # 3. Enforce Monotonically Decreasing Constraint
        # Since the hull is convex (slope is increasing), if the slope becomes positive,
        # it will stay positive. We just need to find the point where it starts rising
        # and flatten it from there.
        
        final_hull = []
        if len(hull) > 0:
            final_hull.append(hull[0])
            
        for i in range(1, len(hull)):
            p_prev = final_hull[-1]
            p_curr = hull[i]
            
            slope = (p_curr[1] - p_prev[1]) / (p_curr[0] - p_prev[0])
            
            if slope > 0:
                # Found a rising segment. 
                # Since we want strictly decreasing (or flat), and we are convex,
                # we effectively stop following the hull here and extend flat.
                # However, we need to cover the full range of r_data.
                # So we add a point at the end of r_data with the same y as p_prev
                final_hull.append((r_data[-1], p_prev[1]))
                break
            else:
                final_hull.append(p_curr)
                
        # If we didn't break, ensure we cover the end (hull usually includes the last point)
        # But if the last point was popped during hull construction (because it was 'above' the line),
        # the hull might end earlier? No, Monotone Chain always includes the last point.
        # But our decreasing check might have stopped early.
        
        # Unzip for interpolation
        r_hull, y_hull = zip(*final_hull)
        
        # Final Interpolation
        bg_valid = np.interp(r_data, r_hull, y_hull)
        
        # User request: "apply smoothing to the convex to make it continues using that know algorithm"
        # Apply Savitzky-Golay filter to smooth the corners
        window_len = 51
        if window_len > len(bg_valid):
            window_len = len(bg_valid)
        if window_len % 2 == 0:
            window_len -= 1
            
        if window_len >= 5:
            bg_valid = savgol_filter(bg_valid, window_length=window_len, polyorder=3)
        
        # Construct full background
        bg_profile = np.zeros_like(intensity)
        bg_profile[start_idx:] = bg_valid
        bg_profile[:start_idx] = bg_init_profile
        
        # Clamp
        bg_profile = np.clip(bg_profile, 0, max_intensity * 1.1)
        
        return bg_profile, start_idx
        
    except Exception as e:
        logger.warning(f"Background fit failed: {e}")
        return np.zeros_like(intensity), start_idx
        

        


        


def baseline_als(y, lam, p, niter=10):
    """
    Asymmetric Least Squares Smoothing (ALS).
    
    Parameters:
    - y: signal data
    - lam: smoothness parameter (10^2 <= lam <= 10^9)
    - p: asymmetry parameter (0.001 <= p <= 0.1)
    - niter: number of iterations
    """
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z.tocsr(), w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

