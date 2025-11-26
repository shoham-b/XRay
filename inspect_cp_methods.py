
import pyFAI.control_points
import inspect

print("Inspecting ControlPoints methods...")
methods = inspect.getmembers(pyFAI.control_points.ControlPoints, predicate=inspect.isfunction)
for name, _ in methods:
    print(name)
