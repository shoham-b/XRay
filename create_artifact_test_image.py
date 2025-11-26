import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_artifact_test_image(filename="test_image_artifacts.png", size=(500, 500), center=(260, 240)):
    """Creates a synthetic diffraction pattern with a known center and artifacts."""
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
    
    # --- Add Artifacts ---
    
    # 1. Beam Stop (Central Block)
    # High intensity block in the center (which becomes dark in inverted image)
    # In raw "intensity" map, this is usually 0 (blocked) or very high (direct beam).
    # Let's assume it's a shadow, so 0 intensity.
    beam_stop_radius = 40
    intensity[r < beam_stop_radius] = 0
    
    # 2. Beam Stop Arm (Line)
    # A line extending from center to edge
    # Let's make it horizontal to the right
    arm_width = 10
    mask_arm = (y > cy - arm_width/2) & (y < cy + arm_width/2) & (x > cx)
    intensity[mask_arm] = 0
    
    # Invert for "film" look
    # Film: 255 - intensity.
    # So 0 intensity (blocked) becomes 255 (white) on film.
    # High intensity (rings) becomes dark.
    
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
    create_artifact_test_image()
