import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_test_image(filename="test_image_adv.png", size=(500, 500), center=(260, 240)):
    """Creates a synthetic diffraction pattern with a known center."""
    h, w = size
    cx, cy = center
    
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Create rings
    intensity = np.zeros((h, w))
    for radius in [50, 100, 150]:
        intensity += 100 * np.exp(-(r - radius)**2 / (2 * 5**2))
        
    # Add background
    intensity += 10
    
    # Add noise
    intensity += np.random.normal(0, 2, (h, w))
    
    # Invert for "film" look (dark rings on light background)
    # But our pipeline inverts it back, so let's make it look like raw data (light rings on dark bg? No, film is negative)
    # Raw film: Dark spots (high intensity) on light background.
    # So we want high values (white) to be background (low intensity) and low values (black) to be peaks (high intensity).
    
    # Let's just create the "intensity" map (high = signal) and then invert it to simulate film.
    # Max intensity ~ 110.
    # Film: 255 - intensity.
    
    film = 255 - intensity
    film = np.clip(film, 0, 255).astype(np.uint8)
    
    # Add a "blue" channel that should be removed
    img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    img_rgb[:, :, 0] = film # R
    img_rgb[:, :, 1] = film # G
    img_rgb[:, :, 2] = 255  # B (noise)
    
    Image.fromarray(img_rgb).save(filename)
    print(f"Created {filename} with center at {center}")

if __name__ == "__main__":
    create_test_image()
