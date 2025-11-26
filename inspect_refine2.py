
import pyFAI.geometryRefinement
import inspect

print("Inspecting refine2...")
sig = inspect.signature(pyFAI.geometryRefinement.GeometryRefinement.refine2)
print(sig)
