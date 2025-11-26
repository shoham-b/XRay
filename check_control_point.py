
import pyFAI
try:
    from pyFAI.control_points import ControlPoint
    print("Imported ControlPoint")
    print(ControlPoint.__init__.__doc__)
except ImportError:
    print("Failed to import ControlPoint")
