
import numpy as np
from PIL import Image
from xray.extension import image_processing as ip

# Create a dummy image with a green background and some bright spots
w, h = 100, 100
img_arr = np.zeros((h, w, 3), dtype=np.uint8)

# Green background (level 50)
img_arr[:, :, 1] = 50

# Bright spots (level 200) - less than 50% of image
img_arr[40:60, 40:60, 1] = 200

# Blue channel (should be removed)
img_arr[:, :, 2] = 100

img = Image.fromarray(img_arr)
img.save("test_green.png")

# Run remove_blue
try:
    img_no_blue = ip.remove_blue(img)
    arr_no_blue = np.array(img_no_blue)
    
    # Check Blue channel (should be 0)
    if np.all(arr_no_blue[:, :, 2] == 0):
        print("Verification: Blue channel removed.")
    else:
        print("Verification Failed: Blue channel not 0.")
        
    # Check Green channel
    # Median should be 50. So background (50) - 50 = 0.
    # Spots (200) - 50 = 150.
    
    g_channel = arr_no_blue[:, :, 1]
    median_val = np.median(g_channel)
    max_val = np.max(g_channel)
    
    print(f"Green Stats after subtraction: Median={median_val}, Max={max_val}")
    
    if median_val == 0 and max_val > 140: # Allow some float/int conversion wiggle room
        print("Verification: Green background successfully subtracted (Median is 0).")
    else:
        print(f"Verification Failed: Green background not removed correctly. Median={median_val}")

except Exception as e:
    print(f"Error: {e}")
