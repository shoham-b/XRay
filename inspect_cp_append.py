
import pyFAI.control_points
import inspect

print("Inspecting ControlPoints.append...")
sig = inspect.signature(pyFAI.control_points.ControlPoints.append)
print(sig)
