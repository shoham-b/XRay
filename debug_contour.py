import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, center_of_mass
import sys

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
    # Fix: No inversion needed as per previous fix
    img_inverted = img_arr
    return img_inverted

def debug_contour(image_path):
    print(f"Debugging contour on {image_path}")
    img = load_image(image_path)
    
    # Smooth
    img_smoothed = gaussian_filter(img, sigma=2)
    
    min_val = np.min(img_smoothed)
    max_val = np.max(img_smoothed)
    p99 = np.percentile(img_smoothed, 99)
    p995 = np.percentile(img_smoothed, 99.5)
    p999 = np.percentile(img_smoothed, 99.9)
    
    print(f"Stats:")
    print(f"  Min: {min_val:.2f}")
    print(f"  Max: {max_val:.2f}")
    print(f"  99.0%: {p99:.2f}")
    print(f"  99.5%: {p995:.2f}")
    print(f"  99.9%: {p999:.2f}")
    
    target_percent = 0.925
    tolerance = 0.025
    
    # Test with Global Max
    target_val = target_percent * max_val
    tol_val = tolerance * max_val
    lower = target_val - tol_val
    upper = target_val + tol_val
    mask = (img_smoothed >= lower) & (img_smoothed <= upper)
    count = np.sum(mask)
    print(f"\nUsing Global Max ({max_val:.2f}):")
    print(f"  Target: {target_val:.2f} (+/- {tol_val:.2f})")
    print(f"  Range: {lower:.2f} - {upper:.2f}")
    print(f"  Pixels Selected: {count}")
    
    # Test with Robust Max (99.5%)
    target_val_r = target_percent * p995
    tol_val_r = tolerance * p995
    lower_r = target_val_r - tol_val_r
    upper_r = target_val_r + tol_val_r
    mask_r = (img_smoothed >= lower_r) & (img_smoothed <= upper_r)
    count_r = np.sum(mask_r)
    print(f"\nUsing Robust Max 99.5% ({p995:.2f}):")
    print(f"  Target: {target_val_r:.2f} (+/- {tol_val_r:.2f})")
    print(f"  Range: {lower_r:.2f} - {upper_r:.2f}")
    print(f"  Pixels Selected: {count_r}")

    if count > 0:
        weights = img_smoothed * mask
        cy, cx = center_of_mass(weights)
        print(f"  CoM (Global): ({cx:.2f}, {cy:.2f})")
        
    if count_r > 0:
        weights_r = img_smoothed * mask_r
        cy_r, cx_r = center_of_mass(weights_r)
        print(f"  CoM (Robust): ({cx_r:.2f}, {cy_r:.2f})")

if __name__ == "__main__":
    debug_contour("data/extension/10_1.jpg")
