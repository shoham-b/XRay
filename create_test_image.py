import numpy as np
from PIL import Image
import os

def create_dummy_diffraction_image(filename="test_image.png", size=(500, 500)):
    # Create a blank image (white background because we invert it later)
    # Actually the code expects a "negative" film where dark = high intensity
    # So we want a light background with dark rings
    
    # Let's make a dark background with light rings (positive) 
    # because the code does: img_inverted = 255.0 - img_arr
    # If input is "negative film" (dark = high intensity), then 
    # img_arr has low values for high intensity.
    # 255 - low = high intensity. Correct.
    
    # So we want to simulate a "negative" film:
    # Background: Light (High pixel values)
    # Peaks: Dark (Low pixel values)
    
    h, w = size
    y, x = np.indices((h, w))
    center = (h // 2, w // 2)
    
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Create a base intensity (light background)
    img_data = np.ones((h, w)) * 200 
    
    # Add some rings (darker)
    # Ring 1 at r=100
    ring1 = np.exp(-((r - 100)**2) / (2 * 5**2)) * 100
    # Ring 2 at r=180
    ring2 = np.exp(-((r - 180)**2) / (2 * 5**2)) * 80
    
    # Subtract rings from background (making them darker)
    img_data -= ring1
    img_data -= ring2
    
    # Clip
    img_data = np.clip(img_data, 0, 255).astype(np.uint8)
    
    # Create RGB image
    img_rgb = np.stack([img_data, img_data, img_data], axis=-1)
    
    # Add some "Blue" noise to test remove_blue
    # Make blue channel different
    img_rgb[:, :, 2] = 255 # Full blue everywhere
    
    img = Image.fromarray(img_rgb)
    img.save(filename)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_dummy_diffraction_image()
