import json
import os

files = [
    "doctor_evaluations.json",
    "video_list.json",
    "patient_symptoms.json",
    "schedules.json",
    "lich_su_tap_luyen.json"
]

for f in files:
    with open(f, 'w', encoding='utf-8') as file:
        json.dump([], file)
    print(f"Cleared {f}")

# Cleanup folders
import shutil
for folder in ["patient_uploads", "temp_frames"]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.makedirs(folder)
        print(f"Reset folder {folder}")
