
import numpy as np
from xray.extension import image_processing as ip

# Create dummy image with a ring
h, w = 200, 200
y, x = np.indices((h, w))
cx, cy = 100, 100
r = np.sqrt((x - cx)**2 + (y - cy)**2)
img = np.exp(-0.5 * (r - 50)**2 / 2**2) # Ring at r=50
img = (img * 255).astype(np.uint8)

# Parameters
pixel_size_mm = 0.1
dist_mm = 100
wavelength_pm = 71.1

print("Testing refine_center_pyfai directly...")
try:
    cx_found, cy_found, radii = ip.refine_center_pyfai(
        img, 
        (100, 100), 
        pixel_size_mm=pixel_size_mm, 
        dist_mm=dist_mm, 
        wavelength_pm=wavelength_pm
    )
    print(f"Direct Result: ({cx_found}, {cy_found})")
except Exception as e:
    print(f"Direct Error: {e}")
    import traceback
    traceback.print_exc()
