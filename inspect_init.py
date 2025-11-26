
import pyFAI.geometryRefinement

import pyFAI.geometryRefinement
import inspect

print("Inspecting GeometryRefinement.__init__...")
sig = inspect.signature(pyFAI.geometryRefinement.GeometryRefinement.__init__)
print(sig)

# Check docstprint(sig)
doc = pyFAI.geometryRefinement.GeometryRefinement.__init__.__doc__
if doc:
    for line in doc.split('\n'):
        print(line)
