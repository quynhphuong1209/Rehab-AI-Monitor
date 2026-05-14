import os
import re

file_path = r'c:\Users\dinhl\Downloads\Rehab-AI-Monitor\app.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find all st.markdown blocks
markdown_blocks = re.findall(r'st\.markdown\("""(.*?)"""', content, re.DOTALL)

for i, block in enumerate(markdown_blocks):
    if 'display: none' in block or 'visibility: hidden' in block or 'stHeader' in block or 'footer' in block:
        print(f"Block {i} contains potential hiding CSS:")
        print(block)
        print("-" * 50)
