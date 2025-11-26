
import pyFAI
import pyFAI.calibration
import inspect

print(f"pyFAI version: {pyFAI.version}")

# Check Calibration class
print("\n--- pyFAI.calibration.Calibration methods ---")
calib = pyFAI.calibration.Calibration(ai=pyFAI.AzimuthalIntegrator())
methods = inspect.getmembers(calib, predicate=inspect.ismethod)
for name, _ in methods:
    if "peak" in name or "refine" in name or "center" in name:
        print(name)

# Check if there's a way to fit circles
print("\n--- pyFAI.geometry methods ---")
import pyFAI.geometry
methods = inspect.getmembers(pyFAI.geometry, predicate=inspect.isfunction)
for name, _ in methods:
    if "fit" in name:
        print(name)
