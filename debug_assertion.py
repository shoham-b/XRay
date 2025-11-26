import numpy as np
from pyFAI.geometryRefinement import GeometryRefinement
from pyFAI.detectors import Detector
from pyFAI.calibrant import Calibrant

# Create test data
pts_2d = np.random.rand(10, 2) * 100
print(f"Data shape: {pts_2d.shape}")
print(f"Data type: {pts_2d.dtype}")
print(f"Data:\n{pts_2d[:3]}")

# Create detector
det = Detector(pixel1=1e-4, pixel2=1e-4)
print(f"\nDetector: {det}")
print(f"Detector pixel1: {det.pixel1}")
print(f"Detector pixel2: {det.pixel2}")

# Try to init with minimal args
print("\n" + "="*50)
print("Attempting to initialize GeometryRefinement...")
print("="*50)

try:
    # Maybe the issue is that data needs to be formatted differently?
    # Let's check if the assertion is about data format
    print(f"data type: {type([pts_2d])}")
    print(f"data[0] type: {type(pts_2d)}")
    print(f"data[0] shape: {pts_2d.shape}")
    print(f"len(data): {len([pts_2d])}")
    
    refiner = GeometryRefinement(
        data=[pts_2d],
        dist=0.1
    )
    print("SUCCESS!")
except AssertionError as e:
    with open('error_details.txt', 'w') as f:
        f.write(f"AssertionError caught: {e}\n")
        f.write(f"Error args: {e.args}\n\n")
        
        # Try to get more info
        import sys
        exc_type, exc_value, exc_tb = sys.exc_info()
        f.write(f"Exception type: {exc_type}\n")
        f.write(f"Exception value: {exc_value}\n\n")
        
        # Print locals at the point of error if possible
        if exc_tb:
            frame = exc_tb.tb_frame
            f.write(f"Frame locals: {list(frame.f_locals.keys())}\n\n")
            
        import traceback
        f.write("Full traceback:\n")
        traceback.print_exc(file=f)
        
    print("Error details written to error_details.txt")
