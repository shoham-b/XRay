
try:
    from pyFAI import geometryRefinement
    print("Imported geometryRefinement")
    print(dir(geometryRefinement))
except ImportError:
    print("Failed to import geometryRefinement")
    try:
        import pyFAI.geometry_refinement
        print("Imported geometry_refinement")
    except ImportError:
        print("Failed to import geometry_refinement")
