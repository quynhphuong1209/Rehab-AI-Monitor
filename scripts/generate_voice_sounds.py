"""Generate Vietnamese TTS voice files for pose feedback (Đúng / Gần đúng / Sai)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUNDS_DIR = os.path.join(ROOT, "sounds")

FILES = {
    "dung.mp3": "Đúng",
    "gan_dung.mp3": "Gần đúng",
    "sai.mp3": "Sai",
}


def main() -> int:
    os.makedirs(SOUNDS_DIR, exist_ok=True)
    try:
        from gtts import gTTS
    except ImportError:
        print("Thieu gTTS. Cai dependencies tu requirements-prod.txt.")
        return 1

    ok = 0
    for filename, text in FILES.items():
        path = os.path.join(SOUNDS_DIR, filename)
        try:
            gTTS(text=text, lang="vi").save(path)
            size = os.path.getsize(path) if os.path.exists(path) else 0
            print(f"OK {filename} size={size}")
            ok += 1
        except Exception as exc:
            print(f"FAIL {filename}: {type(exc).__name__}")
    if ok == len(FILES):
        marker = os.path.join(SOUNDS_DIR, ".voice_tts_ok")
        with open(marker, "w", encoding="utf-8") as mf:
            mf.write("gtts\n")
    return 0 if ok == len(FILES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
