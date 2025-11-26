import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_debug_rings(filename="test_debug_rings.png", size=(300, 200), center=(50, 100), radius=40):
    """
    Creates a synthetic diffraction pattern with a known asymmetric center and radius.
    Size is (Height, Width) = (300, 200) -> Non-square to catch transpose issues.
    Center is (x, y) = (50, 100).
    """
    h, w = size
    cx, cy = center
    
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Create one sharp ring at known radius
    intensity = np.zeros((h, w))
    # Gaussian ring
    intensity += 100 * np.exp(-(r - radius)**2 / (2 * 2**2))
    
    # Add background
    intensity += 10
    
    # Invert for "film" look
    film = 255 - intensity
    film = np.clip(film, 0, 255).astype(np.uint8)
    
    # Add a "blue" channel that should be removed
    img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    img_rgb[:, :, 0] = film # R
    img_rgb[:, :, 1] = film # G
    img_rgb[:, :, 2] = 255  # B
    
    Image.fromarray(img_rgb).save(filename)
    print(f"Created {filename}")
    print(f"Dimensions (H, W): {h}, {w}")
    print(f"Center (x, y): {cx}, {cy}")
    print(f"Ring Radius: {radius}")

if __name__ == "__main__":
    create_debug_rings()
