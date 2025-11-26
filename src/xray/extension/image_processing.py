import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, center_of_mass, label

def remove_blue(image_path):
    # 1. Open image and ensure it is RGB (3 channels) or RGBA (4 channels)
    raw = Image.open(image_path).convert("RGB")
    
    # 2. Convert to NumPy array
    data = np.array(raw)
    
    # 3. Subtract Green Background (Median)
    # Instead of just min(), we use median (50th percentile) to remove the bulk of the background.
    # This assumes the signal (rings) occupies < 50% of the image.
    g = data[:, :, 1]
    g_bg = np.percentile(g, 30)
    print(f"Green Background Subtraction: Removing median value {g_bg}")
    # Use int16 to avoid underflow
    data[:, :, 1] = np.maximum(g.astype(int) - g_bg, 0).astype(np.uint8)
    
    # 4. Set the Blue channel to 0
    # Syntax: [all rows, all columns, channel index 2] (0=R, 1=G, 2=B)
    data[:, :, 2] = 0
    
    # 5. Convert back to Image and return
    return Image.fromarray(data)

def load_and_preprocess_image(image_path):
    """
    Loads an image, converts to grayscale, and inverts it.
    """
    try:
        # Use remove_blue to get the RGB image with Blue=0
        img_no_blue_rgb = remove_blue(image_path)
        # Convert to grayscale for processing
        img = img_no_blue_rgb.convert('L')
    except FileNotFoundError:
        print(f"Error: File {image_path} not found.")
        return None, None, None, None

    img_arr = np.array(img)
    # Invert: Film is negative (dark = high intensity)
    # User request: "OK, the image should be inverted"
    img_inverted = 255 - img_arr
    
    # Create inverted image for saving (visuals might still expect white background?)
    # Actually, for visuals, we usually want bright rings on dark background too.
    img_inverted_pil = Image.fromarray(img_inverted.astype(np.uint8))
    
    return img_arr, img_inverted, img_no_blue_rgb, img_inverted_pil

def find_center(img_inverted):
    """
    Finds the center of the diffraction pattern using the high-intensity contour method.
    """
    # 1. Preprocess
    # Smooth for robustness
    # User request: "use more aggressive bluuring"
    img_smoothed = gaussian_filter(img_inverted, sigma=5)
    
    centers_dict = {}
    
    # 2. Find Center using Exponential Weighted CoM (96% Threshold)
    # User request: "the exponential decay was better"
    try:
        cx_contour, cy_contour, contour_mask = find_center_by_intensity_contour(img_smoothed, threshold_percent=0.92)
        print(f"Exponential Weighted Center (96%): ({cx_contour:.2f}, {cy_contour:.2f})")
        
        centers_dict['ExponentialWeighted'] = (cx_contour, cy_contour)
        centers_dict['ContourMask'] = contour_mask
        
        cx_final, cy_final = cx_contour, cy_contour
        
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

def find_center_by_intensity_contour(img, threshold_percent=0.96, margin=50):
    """
    Finds the center of mass using straight intensity weighting.
    Selects pixels ABOVE the threshold (signal) and keeps only the 
    connected component that contains the MAXIMUM intensity pixel.
    Ignores pixels within 'margin' distance from the edges.
    """
    # 1. Find threshold (robustly)
    threshold = np.percentile(img, threshold_percent * 100)
    
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
        raise ValueError(f"No pixels found above {threshold_percent*100}% intensity (after edge removal)")
        
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
    weights = np.exp(masked_img - max_val)
    
    # Apply mask
    weights = weights * mask
    
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

    # Azimuthal integration (Average intensity at radius r)
    tbin = np.bincount(r_pixels_int.ravel(), img_inverted.ravel())
    nr = np.bincount(r_pixels_int.ravel())
    nr[nr == 0] = 1
    radial_profile = tbin / nr
    
    # Limit to the nearest edge
    # User request: "the max radius shown in the graphs should be the smalles r that the whole circle fits"
    h, w = img_inverted.shape
    max_radius = min(center_x, center_y, w - center_x, h - center_y)
    max_radius_int = int(max_radius)
    
    # Slice the profile
    if max_radius_int < len(radial_profile):
        radial_profile = radial_profile[:max_radius_int]
    
    return radial_profile
