import numpy as np
from pyFAI.geometryRefinement import GeometryRefinement
from pyFAI.detectors import Detector
from pyFAI.calibrant import Calibrant
import traceback
import sys

def test_init(**kwargs):
    """Test GeometryRefinement.__init__ with given kwargs"""
    try:
        print(f"\nTesting with: {list(kwargs.keys())}")
        refiner = GeometryRefinement(**kwargs)
        print("SUCCESS!")
        return True
    except AssertionError as e:
        print(f"AssertionError: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Other error: {e}")
        traceback.print_exc()
        return False

# Create basic components
det = Detector(pixel1=1e-4, pixel2=1e-4)
calibrant = Calibrant(dSpacing=[1.0])

# Test 1: Minimal - just data and dist
print("=" * 50)
print("Test 1: Just data + dist (no detector, no calibrant)")
pts_2d = np.random.rand(10, 2) * 100
test_init(
    data=[pts_2d],
    dist=0.1
)

# Test 2: Add detector
print("=" * 50)
print("Test 2: data + dist + detector")
test_init(
    data=[pts_2d],
    dist=0.1,
    detector=det
)

# Test 3: Add calibrant
print("=" * 50)
print("Test 3: data + dist + detector + calibrant")
test_init(
    data=[pts_2d],
    dist=0.1,
    detector=det,
    calibrant=calibrant
)

# Test 4: With one ring, (n, 3) format
print("=" * 50)
print("Test 4: One ring, (n, 3) format")
pts_3d = np.random.rand(10, 3)  # 10 points, (y, x, I)
pts_3d[:, :2] *= 100  # Scale positions
test_init(
    data=[pts_3d],
    dist=0.1,
    detector=det,
    calibrant=calibrant
)

# Test 5: With wavelength
print("=" * 50)
print("Test 5: With wavelength + poni")
test_init(
    data=[pts_3d],
    dist=0.1,
    poni1=0.05,
    poni2=0.05,
    wavelength=1e-10,
    detector=det,
    calibrant=calibrant
)
