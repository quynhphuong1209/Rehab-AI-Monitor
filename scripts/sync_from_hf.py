# -*- coding: utf-8 -*-
"""
sync_from_hf.py
===============
Script độc lập: Tải kết quả mới nhất (NCV + Bác sĩ) từ HuggingFace Dataset
về và cập nhật vào các file JSON local.

Cách dùng:
    python sync_from_hf.py                          # Dùng token từ .streamlit/secrets.toml hoặc biến môi trường
    python sync_from_hf.py --token hf_xxxx          # Chỉ định token thủ công
    python sync_from_hf.py --dataset quynhphuong1209/Rehab-AI-Monitor-2026-data
    python sync_from_hf.py --dry-run                # Xem trước, không ghi file
    python sync_from_hf.py --file doctor_evaluations.json video_list.json  # Chỉ sync 1 số file
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# ── Cấu hình mặc định ──────────────────────────────────────────────────────
DEFAULT_DATASET_ID = "quynhphuong1209/Rehab-AI-Monitor-2026-data"

# Danh sách file JSON cần đồng bộ (tên file trên HF Dataset → local path)
SYNC_FILES = {
    "doctor_evaluations.json": "database/doctor_evaluations.json",
    "video_list.json":         "database/video_list.json",
    "lich_su_tap_luyen.json":  "database/lich_su_tap_luyen.json",
    "users.json":              "database/users.json",
    "research_data.json":      "database/research_data.json",
    "patient_symptoms.json":   "database/patient_symptoms.json",
    "schedules.json":          "database/schedules.json",
}

# ── Đọc token / dataset_id ─────────────────────────────────────────────────
def load_hf_config():
    """Đọc HF_TOKEN và HF_DATASET_ID theo thứ tự ưu tiên:
    1. Biến môi trường
    2. .streamlit/secrets.toml
    3. Giá trị mặc định (chỉ dataset_id)
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    dataset_id = os.environ.get("HF_DATASET_ID", "").strip()

    # Thử đọc từ .streamlit/secrets.toml
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None

        if tomllib:
            with open(secrets_path, "rb") as f:
                try:
                    secrets = tomllib.load(f)
                    token = token or secrets.get("HF_TOKEN", "").strip()
                    dataset_id = dataset_id or secrets.get("HF_DATASET_ID", "").strip()
                except Exception:
                    pass
        else:
            # Đọc thủ công (không cần thư viện)
            content = secrets_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN") and "=" in line:
                    token = token or line.split("=", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("HF_DATASET_ID") and "=" in line:
                    dataset_id = dataset_id or line.split("=", 1)[1].strip().strip('"').strip("'")

    dataset_id = dataset_id or DEFAULT_DATASET_ID
    return token, dataset_id


# ── Tải 1 file từ HF Dataset qua HTTP ─────────────────────────────────────
def download_hf_file(rel_path: str, token: str, dataset_id: str, timeout=60) -> bytes | None:
    """Tải nội dung file từ HuggingFace Dataset về dưới dạng bytes."""
    import urllib.parse
    try:
        import requests
    except ImportError:
        print("  ⚠️  Cần cài đặt requests: pip install requests")
        return None

    rel_norm = rel_path.replace("\\", "/")
    rel_enc  = urllib.parse.quote(rel_norm, safe="/")
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{rel_enc}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 404:
            print(f"  ⚠️  Không tìm thấy trên HF: {rel_path}")
            return None
        if resp.status_code in (401, 403):
            print(f"  ❌  Không có quyền truy cập (HTTP {resp.status_code}) — kiểm tra HF_TOKEN")
            return None
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"  ❌  Lỗi kết nối khi tải {rel_path}: {e}")
        return None


# ── Merge / cập nhật JSON ──────────────────────────────────────────────────
def merge_json_list(local_path: str, remote_data: list, key_fn=None, label="bản ghi") -> dict:
    """
    Merge danh sách JSON từ remote vào file local.
    - Nếu local chưa có → ghi thẳng remote.
    - Nếu local đã có  → thêm các bản ghi mới (theo key_fn), giữ nguyên bản cũ.
    Trả về dict {"added": int, "total": int}.
    """
    # Đọc local
    local_data = []
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                local_data = json.load(f)
            if not isinstance(local_data, list):
                local_data = []
        except Exception:
            local_data = []

    if not key_fn:
        # Không có key → ghi thẳng phiên bản mới nhất (remote thường mới hơn)
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(remote_data, f, ensure_ascii=False, indent=2)
        return {"added": len(remote_data), "total": len(remote_data)}

    # Xây index từ local
    local_keys = {}
    for item in local_data:
        k = key_fn(item)
        if k:
            local_keys[k] = item

    added = 0
    for item in remote_data:
        k = key_fn(item)
        if k and k not in local_keys:
            local_data.append(item)
            local_keys[k] = item
            added += 1
        elif k and k in local_keys:
            # Cập nhật nếu remote có trường mới hơn (không ghi đè toàn bộ)
            existing = local_keys[k]
            for field, val in item.items():
                if field not in existing or (val and not existing.get(field)):
                    existing[field] = val

    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(local_data, f, ensure_ascii=False, indent=2)

    return {"added": added, "total": len(local_data)}


# ── Key functions cho từng loại file ─────────────────────────────────────
def _eval_key(e):
    """Key duy nhất cho doctor_evaluations: patient + video + thời gian."""
    return f"{e.get('patient_username')}|{e.get('video_name')}|{e.get('time')}|{e.get('doctor_username')}"

def _video_key(v):
    """Key duy nhất cho video_list."""
    return f"{v.get('username')}|{v.get('video_name')}"

def _history_key(h):
    return f"{h.get('username')}|{h.get('video_name')}|{h.get('bai_tap')}"

def _user_key(u):
    if isinstance(u, dict):
        return u.get("username")
    return None


FILE_KEY_FN = {
    "doctor_evaluations.json": _eval_key,
    "video_list.json":         _video_key,
    "lich_su_tap_luyen.json":  _history_key,
    "users.json":              None,   # ghi thẳng (ít thay đổi)
    "research_data.json":      None,
    "patient_symptoms.json":   None,
    "schedules.json":          None,
}


# ── Hàm chính ──────────────────────────────────────────────────────────────
def sync_from_hf(
    token: str,
    dataset_id: str,
    files: list[str] | None = None,
    dry_run: bool = False,
    backup: bool = True,
):
    """
    Tải và merge dữ liệu từ HF Dataset về local.

    Args:
        token:      HF token
        dataset_id: ID của dataset trên HuggingFace
        files:      Danh sách tên file cần sync (None = tất cả)
        dry_run:    Nếu True, chỉ in ra kết quả, không ghi file
        backup:     Nếu True, backup file local trước khi ghi
    """
    print(f"\n{'='*60}")
    print(f"  📥  SYNC KẾT QUẢ TỪ HUGGINGFACE DATASET")
    print(f"{'='*60}")
    print(f"  Dataset : {dataset_id}")
    print(f"  Token   : {'✅ Có' if token else '⚠️  Không có (chỉ đọc được dataset public)'}")
    print(f"  Mode    : {'🔍 DRY-RUN (chỉ xem)' if dry_run else '💾 GHI FILE THẬT'}")
    print()

    target_files = files or list(SYNC_FILES.keys())

    for hf_name in target_files:
        local_rel = SYNC_FILES.get(hf_name)
        if not local_rel:
            print(f"  ⚠️  Không biết đường dẫn local cho: {hf_name}, bỏ qua.")
            continue

        local_path = os.path.normpath(local_rel)
        print(f"  📄  {hf_name}")

        # Tải từ HF
        raw = download_hf_file(hf_name, token, dataset_id)
        if raw is None:
            print(f"       → Bỏ qua (không tải được).\n")
            continue

        # Parse JSON
        try:
            remote_data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            print(f"       → ❌ Parse JSON lỗi: {e}\n")
            continue

        # Thống kê
        if isinstance(remote_data, list):
            print(f"       Trên HF : {len(remote_data)} bản ghi")
        elif isinstance(remote_data, dict):
            print(f"       Trên HF : dict ({len(remote_data)} keys)")

        if dry_run:
            print(f"       [DRY-RUN] Không ghi file.\n")
            continue

        # Backup file local cũ
        if backup and os.path.exists(local_path):
            bak_path = local_path + ".bak"
            shutil.copy2(local_path, bak_path)

        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

        # Merge
        if isinstance(remote_data, list):
            key_fn = FILE_KEY_FN.get(hf_name)
            result = merge_json_list(local_path, remote_data, key_fn)
            print(f"       Local   : {result['total']} bản ghi (mới thêm: +{result['added']})")
        else:
            # Với dict (ví dụ users.json có thể là dict) → ghi thẳng
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(remote_data, f, ensure_ascii=False, indent=2)
            print(f"       → Ghi thẳng (dict).")

        print(f"       ✅ Đã cập nhật: {local_path}\n")

    print(f"{'='*60}")
    print(f"  ✅  Đồng bộ hoàn tất lúc {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")
    print(f"{'='*60}\n")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Đồng bộ dữ liệu NCV + Bác sĩ từ HuggingFace Dataset về local JSON."
    )
    parser.add_argument("--token",   type=str, default=None, help="HF Token (ghi đè biến môi trường)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset ID trên HuggingFace")
    parser.add_argument("--file",    type=str, nargs="+",   help="Chỉ sync các file cụ thể (VD: doctor_evaluations.json video_list.json)")
    parser.add_argument("--dry-run", action="store_true",   help="Chỉ xem, không ghi file")
    parser.add_argument("--no-backup", action="store_true", help="Không tạo file .bak")
    args = parser.parse_args()

    # Đọc config
    env_token, env_dataset = load_hf_config()
    token      = args.token   or env_token
    dataset_id = args.dataset or env_dataset

    if not token:
        print("\n⚠️  Chưa có HF_TOKEN!")
        print("   Cách 1: Thêm HF_TOKEN=hf_xxxx vào file .streamlit/secrets.toml")
        print("   Cách 2: Chạy:  set HF_TOKEN=hf_xxxx  (Windows) rồi chạy lại script")
        print("   Cách 3: Truyền:  python sync_from_hf.py --token hf_xxxx\n")
        confirm = input("Vẫn thử tải (dataset public không cần token)? [y/N] ").strip().lower()
        if confirm != "y":
            sys.exit(0)

    sync_from_hf(
        token      = token,
        dataset_id = dataset_id,
        files      = args.file,
        dry_run    = args.dry_run,
        backup     = not args.no_backup,
    )
