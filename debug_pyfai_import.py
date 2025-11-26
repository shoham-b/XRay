
import pyFAI
print(f"pyFAI file: {pyFAI.__file__}")
print(f"pyFAI dir: {dir(pyFAI)}")
try:
    from pyFAI import calibration
    print("Imported calibration")
except ImportError as e:
    print(f"Failed to import calibration: {e}")
