import re
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "scripts":
    root_dir = os.path.dirname(script_dir)
else:
    root_dir = script_dir

file_path = os.path.join(root_dir, 'app.py')

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern 1: xaxis=dict(title="...", title_font=dict(...), ...)
content = re.sub(
    r'(xaxis|yaxis)=dict\(\s*title=([^,]+),\s*title_font=(dict\([^)]+\)),',
    r'\1=dict(title=dict(text=\2, font=\3),',
    content
)

# Pattern 2: fig.update_xaxes(title_text="...", title_font=dict(...), ...)
content = re.sub(
    r'title_text=([^,]+),\s*title_font=(dict\([^)]+\)),',
    r'title=dict(text=\1, font=\2),',
    content
)

# Pattern 3: Case where gridcolor or other props follow title_font
content = re.sub(
    r'title=([^,]+),\s*title_font=(dict\([^)]+\))',
    r'title=dict(text=\1, font=\2)',
    content
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Successfully updated {file_path} with Plotly 6.0 compatible schema.")
