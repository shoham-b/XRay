
try:
    import pyFAI.goniometer
    print("Imported goniometer")
    print(dir(pyFAI.goniometer))
except ImportError:
    print("Failed to import goniometer")
