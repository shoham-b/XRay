
import pyFAI.calibrant
import inspect

print("Inspecting Calibrant.__init__...")
sig = inspect.signature(pyFAI.calibrant.Calibrant.__init__)
print(sig)
print(pyFAI.calibrant.Calibrant.__init__.__doc__)
