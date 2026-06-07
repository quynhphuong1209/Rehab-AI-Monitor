import json
import os
import shutil

# Get root directory relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
# If running from inside 'scripts' folder, root is parent, otherwise it's script_dir
if os.path.basename(script_dir) == "scripts":
    root_dir = os.path.dirname(script_dir)
else:
    root_dir = script_dir

db_dir = os.path.join(root_dir, "database")

# Files to reset (stored in database directory)
db_files = [
    "doctor_evaluations.json",
    "video_list.json",
    "patient_symptoms.json",
    "schedules.json",
    "lich_su_tap_luyen.json"
]

# Reset database files
if not os.path.exists(db_dir):
    os.makedirs(db_dir, exist_ok=True)

for f_name in db_files:
    f_path = os.path.join(db_dir, f_name)
    with open(f_path, 'w', encoding='utf-8') as file:
        json.dump([], file)
    print(f"Cleared {f_path}")

# Reset directories (stored in root directory)
media_folders = ["patient_uploads", "temp_frames"]
for folder in media_folders:
    folder_path = os.path.join(root_dir, folder)
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            print(f"Error removing {folder_path}: {e}")
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Reset folder {folder_path}")
    except Exception as e:
        print(f"Error creating {folder_path}: {e}")
