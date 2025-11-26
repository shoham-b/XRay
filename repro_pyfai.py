
import numpy as np
from pyFAI.geometryRefinement import GeometryRefinement
from pyFAI.detectors import Detector
from pyFAI.calibrant import Calibrant

print("Creating dummy data...")
# 1 ring, 10 points, (y, x, I)
pts = np.random.rand(10, 3)
data = [pts]

d_spacings = [1.0] # 1 ring
calibrant = Calibrant(dSpacing=d_spacings)

det = Detector(pixel1=1e-4, pixel2=1e-4)

print("Initializing GeometryRefinement...")
try:
    refiner = GeometryRefinement(
        data=data,
        dist=0.1,
        poni1=0.1,
        poni2=0.1,
        pixel1=1e-4,
        pixel2=1e-4,
        wavelength=1e-10,
        detector=det,
        calibrant=calibrant
    )
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
