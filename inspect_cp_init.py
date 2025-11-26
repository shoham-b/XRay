
import pyFAI.control_points
import inspect

print("Inspecting ControlPoints.__init__...")
sig = inspect.signature(pyFAI.control_points.ControlPoints.__init__)
print(sig)
print(pyFAI.control_points.ControlPoints.__init__.__doc__)
