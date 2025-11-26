import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, center_of_mass
import sys
import os

# Mock the necessary parts of image_processing
def remove_blue(image_path):
    raw = Image.open(image_path).convert("RGB")
    data = np.array(raw)
    g = data[:, :, 1]
    g_bg = np.percentile(g, 50)
    data[:, :, 1] = np.maximum(g.astype(int) - g_bg, 0).astype(np.uint8)
    data[:, :, 2] = 0
    return Image.fromarray(data)

def load_image(image_path):
    img_no_blue_rgb = remove_blue(image_path)
    img = img_no_blue_rgb.convert('L')
    img_arr = np.array(img)
    # img_inverted = 255.0 - img_arr
    return img_arr

def test_masking(image_path):
    print(f"Testing masking on {image_path}")
    img = load_image(image_path)
    h, w = img.shape
    
    # Smooth
    img_smoothed = gaussian_filter(img, sigma=2)
    
    # CoM guess
    border = 50
    mask = np.zeros_like(img_smoothed, dtype=bool)
    mask[border:h-border, border:w-border] = True
    masked_img = img_smoothed * mask
    threshold = np.percentile(masked_img[mask], 99.5)
    binary_img = masked_img > threshold
    weights = masked_img * binary_img
    cy_com, cx_com = center_of_mass(weights)
    start_center = (cx_com, cy_com)
    print(f"Start Center: {start_center}")
    
    # Masking Logic
    y_grid, x_grid = np.indices((h, w))
    cx_start, cy_start = start_center
    r_from_start = np.sqrt((x_grid - cx_start)**2 + (y_grid - cy_start)**2)
    
    r_int_start = r_from_start.astype(int)
    max_r_start = int(r_int_start.max())
    tbin_start = np.bincount(r_int_start.ravel(), weights=img.ravel(), minlength=max_r_start+1)
    nr_start = np.bincount(r_int_start.ravel(), minlength=max_r_start+1)
    nr_start[nr_start == 0] = 1
    profile_start = tbin_start / nr_start
    
    profile_start_smooth = gaussian_filter(profile_start, sigma=2)
    
    max_intensity = np.max(profile_start_smooth)
    upper_thresh = 0.8 * max_intensity
    lower_thresh = 0.4 * max_intensity
    
    print(f"Max Intensity: {max_intensity:.2f}")
    print(f"Profile Min: {np.min(profile_start_smooth):.2f}")
    print(f"Profile Mean: {np.mean(profile_start_smooth):.2f}")
    print(f"Profile Sample (0-100:10): {profile_start_smooth[0:100:10]}")
    print(f"Band: {lower_thresh:.2f} - {upper_thresh:.2f}")
    
    valid_radii_indices = np.where((profile_start_smooth <= upper_thresh) & 
                                   (profile_start_smooth >= lower_thresh))[0]
    
    print(f"Found {len(valid_radii_indices)} valid radii.")
    if len(valid_radii_indices) > 0:
        print(f"Min Radius: {valid_radii_indices.min()}")
        print(f"Max Radius: {valid_radii_indices.max()}")
        print(f"Sample indices: {valid_radii_indices[::10]}")
    else:
        print("NO VALID RADII FOUND!")

if __name__ == "__main__":
    test_masking("data/extension/10_1.jpg")
