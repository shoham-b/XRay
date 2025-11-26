import sys
import os
# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from xray.extension.analysis import process_diffraction_image

try:
    print("Starting analysis...")
    result = process_diffraction_image("test_image.png", 55.0, 75.0, 20.0)
    print("Analysis complete.")
    print("Result keys:", result.keys())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
