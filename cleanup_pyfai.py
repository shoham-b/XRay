"""Clean up image_processing.py by removing leftover pyFAI code"""

with open('src/xray/extension/image_processing.py', 'r') as f:
    lines = f.readlines()

# Remove lines 387-547 (0-indexed: 386-546)
# These contain the orphaned pyFAI function definition
clean_lines = lines[:386] + lines[547:]

with open('src/xray/extension/image_processing.py', 'w') as f:
    f.writelines(clean_lines)

print(f"Removed {547-386} lines of leftover pyFAI code")
print(f"New file has {len(clean_lines)} lines")
