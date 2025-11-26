
import numpy as np
import matplotlib.pyplot as plt
from xray.extension import viz

# Dummy data
img_arr = np.zeros((200, 200))
center_x, center_y = 100, 100
peak_radii_pixels = [50, 5000] # 5000 should be filtered out
base_name = "test_viz"
output_dir = "."
centers_dict = {
    'CoM': (90, 90),
    'Line': (95, 95),
    'Residual': (98, 98),
    'Ring': (100, 100)
}

# Run plot function
try:
    save_path = viz.plot_center_on_image(img_arr, center_x, center_y, peak_radii_pixels, base_name, output_dir, centers_dict)
    print(f"Success: {save_path}")
except Exception as e:
    print(f"Error: {e}")
