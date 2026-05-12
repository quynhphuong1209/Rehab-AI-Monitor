import os

search_term = "current_theme"
file_path = r"c:\Users\dinhl\Downloads\Rehab-AI-Monitor\app.py"

with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if search_term in line:
            print(f"{i}: {line.strip()}")
