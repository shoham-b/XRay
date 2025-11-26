import sys
import os
from xray.extension.analysis import process_diffraction_image

# Redirect stdout to capture output
sys.stdout = open('debug_output.txt', 'w')

def main():
    image_path = "test_debug_rings.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return

    print(f"Running analysis on {image_path}...")
    try:
        result = process_diffraction_image(
            image_path, 
            phys_w_mm=55.0, 
            phys_h_mm=75.0, 
            distance_L_mm=200.0, 
            wavelength_pm=71.1
        )
        print("Analysis completed.")
        print(result)
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
