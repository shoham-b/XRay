import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, center_of_mass, label, median_filter, map_coordinates
from scipy.optimize import minimize

def extract_analysis_channel(image_path):
    # 1. Open image and ensure it is RGB
    raw = Image.open(image_path).convert("RGB")
    
    # 2. Convert to NumPy array
    data = np.array(raw)
    
    # 3. Extract Channels
    # User request: "80% blue and 20% green"
    blue_channel = data[:, :, 2].astype(float)
    green_channel = data[:, :, 1].astype(float)
    
    combined_channel = 0.8 * blue_channel + 0.2 * green_channel
    
    # Clip to valid range and convert to uint8
    combined_channel = np.clip(combined_channel, 0, 255).astype(np.uint8)
    
    # 4. Return as Grayscale Image
    return Image.fromarray(combined_channel, mode='L')

def load_and_preprocess_image(image_path):
    """
    Loads an image, extracts the analysis channel (80% Blue, 20% Green), and inverts it.
    """
    try:
        # Use extract_analysis_channel to get the combined channel as grayscale
        img = extract_analysis_channel(image_path)
        # It's already grayscale ('L')
    except FileNotFoundError:
        print(f"Error: File {image_path} not found.")
        return None, None, None, None

    img_arr = np.array(img)
    # Invert: Film is negative (dark = high intensity)
    # User request: "OK, the image should be inverted"
    img_inverted = 255 - img_arr
    
    # Create inverted image for saving
    img_inverted_pil = Image.fromarray(img_inverted.astype(np.uint8))
    
    # We return 'img' as the 3rd argument (replacing the old img_no_blue_rgb)
    # It is now the "preprocessed" image (Combined Channel)
    return img_arr, img_inverted, img, img_inverted_pil

def find_center_optimization(img, initial_guess):
    """
    Refines the center by minimizing the variance along concentric rings (Angular Variance).
    Uses a polar transform sampled via bilinear interpolation for smoothness.
    """
    h, w = img.shape
    
    # Pre-calculate grid for polar transform (normalized)
    # We'll sample a fixed number of radii and angles
    n_radii = min(h, w) // 2
    n_angles = 360
    
    # r and theta coordinates
    r = np.linspace(0, n_radii, n_radii)
    theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    
    r_grid, theta_grid = np.meshgrid(r, theta)
    
    def objective(center):
        cx, cy = center
        # Penalty for going out of bounds
        if not (0 <= cx < w and 0 <= cy < h):
            return 1e9
            
        # Convert polar to cartesian based on CURRENT center
        x_sample = cx + r_grid * np.cos(theta_grid)
        y_sample = cy + r_grid * np.sin(theta_grid)
        
        # Sample image using bilinear interpolation (order=1)
        # mode='constant', cval=0 handles out of bounds
        polar_img = map_coordinates(img, [y_sample, x_sample], order=1, mode='constant', cval=0.0)
        
        # Calculate Radial Profile (Mean along theta)
        # Ideally, for a centered ring, intensity is constant along theta.
        # We want to maximize the "energy" or "contrast" of the radial profile.
        # If rings are aligned, the peaks in the radial profile are highest.
        # So we maximize sum(profile^2).
        radial_profile = np.mean(polar_img, axis=0)
        
        # We want to MAXIMIZE energy, so minimize negative energy
        energy = np.sum(radial_profile**2)
        return -energy

    # Run optimization
    # Nelder-Mead is robust. Initial simplex size might need adjustment.
    result = minimize(objective, initial_guess, method='Nelder-Mead', tol=0.1, options={'maxiter': 50, 'xatol': 0.1, 'fatol': 0.1})
    
    return result.x[0], result.x[1]

def find_center(img_inverted):
    """
    Finds the center of the diffraction pattern using the high-intensity contour method,
    followed by Radial Symmetry Optimization.
    """
    # 1. Preprocess
    
    # Hot Pixel Filter (User request: "sharply differenct from it's average sourounding intensity")
    # We apply this BEFORE smoothing to remove single pixel spikes.
    
    # Calculate local median (3x3 neighborhood)
    local_median = median_filter(img_inverted, size=3)
    
    # Identify hot/cold pixels: significantly different from median
    # User request: "filter both if much brigher or much darker"
    
    diff = img_inverted.astype(float) - local_median.astype(float)
    
    # Threshold: > 50% deviation from median AND > 20 units absolute difference
    # We use absolute difference to catch both bright spikes and dark holes.
    mask_outliers = (np.abs(diff) > 0.5 * local_median) & (np.abs(diff) > 20)
    
    img_filtered = img_inverted.copy()
    img_filtered[mask_outliers] = local_median[mask_outliers]
    
    # Smooth for robustness
    # User request: "use more aggressive bluuring"
    img_smoothed = gaussian_filter(img_filtered, sigma=5)
    
    centers_dict = {}
    
    # 2. Find Initial Guess using Exponential Weighted CoM (Absolute Threshold 170)
    # User request: "instead of taking the values as percentage of max intensity, take up to 170 of absolte intensity"
    try:
        cx_contour, cy_contour, contour_mask = find_center_by_intensity_contour(img_smoothed, threshold_value=80)
        print(f"Initial Guess (CoM): ({cx_contour:.2f}, {cy_contour:.2f})")
        
        centers_dict['InitialGuess'] = (cx_contour, cy_contour)
        centers_dict['ContourMask'] = contour_mask
        
        # 3. Refine using Radial Symmetry Optimization
        # User request: "there must be some know good algrothim"
        print("Refining center using Radial Symmetry Optimization...")
        cx_opt, cy_opt = find_center_optimization(img_smoothed, (cx_contour, cy_contour))
        print(f"Optimized Center: ({cx_opt:.2f}, {cy_opt:.2f})")
        
        centers_dict['Optimized'] = (cx_opt, cy_opt)
        
        cx_final, cy_final = cx_opt, cy_opt
        
    except Exception as e:
        print(f"Error finding center: {e}")
        h, w = img_inverted.shape
        cx_final, cy_final = w / 2.0, h / 2.0
        centers_dict['Fallback'] = (cx_final, cy_final)

    print(f"Final Center: ({cx_final:.2f}, {cy_final:.2f})")
    
    # Calculate ring radii for visualization/analysis
    try:
        profile = calculate_radial_profile(img_inverted, cx_final, cy_final)
        profile_smoothed = gaussian_filter(profile, sigma=2)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(profile_smoothed, prominence=10, distance=10)
        ring_radii = peaks
    except:
        ring_radii = []

    return int(cx_final), int(cy_final), ring_radii, centers_dict

def find_center_by_intensity_contour(img, threshold_value=170, margin=50):
    """
    Finds the center of mass using straight intensity weighting.
    Selects pixels ABOVE the threshold (signal) and keeps only the 
    connected component that contains the MAXIMUM intensity pixel.
    Ignores pixels within 'margin' distance from the edges.
    """
    # 1. Find threshold (Absolute)
    # User request: "take up to 170 of absolte intensity"
    threshold = threshold_value
    
    # 2. Create mask (pixels ABOVE threshold - i.e., signal)
    mask = img >= threshold
    
    # 3. Remove Edges
    # User request: "remove the edges, the contour should not be on the edges"
    h, w = img.shape
    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False
    
    if np.sum(mask) == 0:
        raise ValueError(f"No pixels found above intensity {threshold_value} (after edge removal)")
        
    # 4. Filter for Component containing Max Intensity
    # "all of the points... should be together... close to the max intensity points"
    labeled_array, num_features = label(mask)
    
    if num_features > 0:
        # Find the location of the max intensity in the MASKED image
        # We must only look for max intensity within the valid mask region
        # Otherwise we might pick a bright spot on the edge that we just masked out
        masked_img = img.copy()
        masked_img[~mask] = 0 # Set non-mask pixels to 0
        
        max_ind = np.argmax(masked_img)
        max_pos = np.unravel_index(max_ind, img.shape)
        
        # Get the label at the max intensity position
        target_label = labeled_array[max_pos]
        
        if target_label == 0:
            # This shouldn't happen if we searched in masked_img, but safety first
            print("Warning: Max intensity pixel not in mask. Falling back to largest component.")
            counts = np.bincount(labeled_array.ravel())
            counts[0] = 0
            target_label = np.argmax(counts)
            
        # Update mask to only include the target component
        mask = (labeled_array == target_label)
        
    # 5. Filter Low Density Regions
    # User request: "remove regimes of low selected pixel densitity"
    # We calculate the local density of selected pixels.
    # size=20 means a 20x20 window.
    from scipy.ndimage import uniform_filter
    density = uniform_filter(mask.astype(float), size=20)
    
    # Threshold density. 0.1 means at least 10% of the 20x20 window is filled.
    # This removes sparse, stringy, or isolated regions.
    mask = mask & (density > 0.5)
    
    if np.sum(mask) == 0:
         raise ValueError("No pixels left after density filtering")

    # 6. Calculate CoM with Exponential Weighting
    # User request: "the exponential decay was better" (Reverting from Inv R^2)
    # We use exp(img) to give massive weight to the peak.
    # To avoid potential overflow (though exp(255) fits in float64), we subtract max first.
    # weights = exp(img - max_img)
    # This makes the max intensity have weight 1, and lower intensities decay exponentially.
    
    # We only care about weights within the mask
    masked_img = img.astype(float)
    
    # Shift values so max is 0 (avoids overflow)
    max_val = np.max(masked_img[mask])
    # We multiply by 2 to make the decay faster, prioritizing the peak.
    weights = np.exp((masked_img - max_val) * 2)
    
    # Apply mask
    weights = weights * mask

    # Calculate Center of Mass
    cy, cx = center_of_mass(weights)
    
    return cx, cy, mask

def calculate_radial_profile(img_inverted, center_x, center_y):
    """
    Calculates the radial profile (azimuthal integration).
    Limits the profile to the radius where full circles fit within the image
    (distance to the nearest edge).
    """
    y, x = np.indices(img_inverted.shape)
    r_pixels = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    r_pixels_int = r_pixels.astype(int)

    # Sector-Based Max Projection
    # 1. Calculate angles
    theta = np.arctan2(y - center_y, x - center_x) # -pi to pi
    
    # 2. Define sectors (e.g., 10 degrees = pi/18)
    n_sectors = 36
    sector_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    
    # Digitize theta to find which sector each pixel belongs to
    sector_indices = np.digitize(theta, sector_edges) - 1
    sector_indices = np.clip(sector_indices, 0, n_sectors - 1)
    
    # 3. Aggregate by (radius, sector)
    # We can use a 2D histogram or similar, but since we need mean, let's use pandas or pure numpy loop
    # Pure numpy loop over radii is likely fastest given the image size
    
    # Flatten arrays
    r_flat = r_pixels_int.ravel()
    s_flat = sector_indices.ravel()
    i_flat = img_inverted.ravel()
    
    # Sort by radius for efficiency
    sort_idx = np.argsort(r_flat)
    r_sorted = r_flat[sort_idx]
    s_sorted = s_flat[sort_idx]
    i_sorted = i_flat[sort_idx]
    
    unique_r, unique_indices, unique_counts = np.unique(r_sorted, return_index=True, return_counts=True)
    
    max_r_val = unique_r[-1]
    radial_profile = np.zeros(max_r_val + 1)
    
    for r_val, start_idx, count in zip(unique_r, unique_indices, unique_counts):
        # Get data for this radius
        s_r = s_sorted[start_idx : start_idx + count]
        i_r = i_sorted[start_idx : start_idx + count]
        
        # Calculate mean intensity per sector
        # We can use bincount for sum and count
        sector_sums = np.bincount(s_r, weights=i_r, minlength=n_sectors)
        sector_counts = np.bincount(s_r, minlength=n_sectors)
        
        # Avoid division by zero
        valid_sectors = sector_counts > 0
        
        if np.any(valid_sectors):
            sector_means = np.zeros(n_sectors)
            sector_means[valid_sectors] = sector_sums[valid_sectors] / sector_counts[valid_sectors]
            
            # Take the MAX of the sector means
            # This ensures that if a ring is present in ANY sector, we detect it.
            radial_profile[r_val] = np.max(sector_means)
        else:
            radial_profile[r_val] = 0
            
    # Limit to the nearest edge
    # User request: "the max radius shown in the graphs should be the smalles r that the whole circle fits"
    h, w = img_inverted.shape
    max_radius = min(center_x, center_y, w - center_x, h - center_y)
    max_radius_int = int(max_radius)
    
    # Slice the profile
    if max_radius_int < len(radial_profile):
        radial_profile = radial_profile[:max_radius_int]
    
    return radial_profile
