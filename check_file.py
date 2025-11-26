import sys

# Read the file
with open('src/xray/extension/image_processing.py', 'r') as f:
    lines = f.readlines()

# Find the problematic section
# Looking for the stray docstring starting around line 387
# and ending at the calculate_radial_profile function

# Find where find_center ends (should be around line 386)
# and where calculate_radial_profile starts (should be around line 549)

for i, line in enumerate(lines[380:400], start=381):
    print(f"{i}: {line}", end='')

print("\n" + "="*50)
print("Lines around calculate_radial_profile:")
for i, line in enumerate(lines[545:555], start=546):
    print(f"{i}: {line}", end='')
