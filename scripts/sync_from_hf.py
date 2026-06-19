# -*- coding: utf-8 -*-
"""
sync_from_hf.py
===============
Script độc lập: Tải kết quả mới nhất (NCV + Bác sĩ) từ HuggingFace Dataset
về và cập nhật vào các file JSON local.

Cách dùng:
    python sync_from_hf.py                          # Dùng token từ .streamlit/secrets.toml hoặc biến môi trường
    python sync_from_hf.py --token <HF_TOKEN>       # Chỉ định token thủ công
    python sync_from_hf.py --dataset <owner/dataset>
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.json_store import read_json, write_json
from cloud.hf_sync import download_dataset_file_bytes

# Danh sách file JSON cần đồng bộ (tên file trên HF Dataset → local path)
SYNC_FILES = {
    "doctor_evaluations.json": "database/doctor_evaluations.json",
    "video_list.json":         "database/video_list.json",
    "lich_su_tap_luyen.json":  "database/lich_su_tap_luyen.json",
    "research_data.json":      "database/research_data.json",
    "patient_symptoms.json":   "database/patient_symptoms.json",
    "schedules.json":          "database/schedules.json",
}
OPTIONAL_SYNC_FILES = {
    "users.json":              "database/users.json",
}

# ── Đọc token / dataset_id ─────────────────────────────────────────────────
def load_hf_config():
    """Đọc HF_TOKEN và HF_DATASET_ID theo thứ tự ưu tiên:
    1. Biến môi trường
    2. .streamlit/secrets.toml
    3. Không có mặc định an toàn: dataset_id phải được cấu hình rõ ràng
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
                except (OSError, tomllib.TOMLDecodeError) as exc:
                    print(f"  ⚠️  Không đọc được secrets.toml: {exc}")
        else:
            # Đọc thủ công (không cần thư viện)
            content = secrets_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN") and "=" in line:
                    token = token or line.split("=", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("HF_DATASET_ID") and "=" in line:
                    dataset_id = dataset_id or line.split("=", 1)[1].strip().strip('"').strip("'")

    return token, dataset_id


# ── Tải 1 file từ HF Dataset qua HTTP ─────────────────────────────────────
def download_hf_file(rel_path: str, token: str, dataset_id: str, timeout=60) -> bytes | None:
    """Tải nội dung file từ HuggingFace Dataset về dưới dạng bytes."""
    raw, err = download_dataset_file_bytes(
        rel_path,
        token=token,
        dataset_id=dataset_id,
        timeout=timeout,
    )
    if raw is not None:
        return raw
    if err and str(err).startswith("Chưa có trên Dataset:"):
        print(f"  ⚠️  Không tìm thấy trên HF: {rel_path}")
    elif err == "Token không có quyền tải file từ Dataset.":
        print("  ❌  Không có quyền truy cập — kiểm tra HF_TOKEN")
    elif err:
        print(f"  ❌  Lỗi kết nối khi tải {rel_path}: {err}")
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
    local_data = read_json(local_path, [])
    if not isinstance(local_data, list):
        local_data = []

    if not key_fn:
        # Không có key → ghi thẳng phiên bản mới nhất (remote thường mới hơn)
        write_json(local_path, remote_data, indent=2)
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

    write_json(local_path, local_data, indent=2)

    return {"added": added, "total": len(local_data)}


def backup_local_file(local_path: str) -> str | None:
    if not os.path.exists(local_path):
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak_path = f"{local_path}.{stamp}.bak"
    shutil.copy2(local_path, bak_path)
    return bak_path


def merge_users_dict(local_path: str, remote_users: dict) -> dict:
    loaded = read_json(local_path, {})
    local_users = loaded if isinstance(loaded, dict) else {}

    protected_fields = {"password", "role", "hash_version", "must_change_password"}
    added = 0
    updated = 0
    for username, remote_record in (remote_users or {}).items():
        if not isinstance(remote_record, dict):
            continue
        if username not in local_users or not isinstance(local_users.get(username), dict):
            local_users[username] = remote_record
            added += 1
            continue

        local_record = local_users[username]
        for field, value in remote_record.items():
            if field in protected_fields:
                continue
            if field not in local_record or (value and not local_record.get(field)):
                local_record[field] = value
                updated += 1

    write_json(local_path, local_users, indent=2)

    return {"added": added, "updated": updated, "total": len(local_users)}


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
    "users.json":              None,
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
    include_users: bool = False,
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

    sync_files = dict(SYNC_FILES)
    if include_users:
        sync_files.update(OPTIONAL_SYNC_FILES)
    target_files = files or list(sync_files.keys())
    if "users.json" in target_files and not include_users:
        print("  ⚠️  Bỏ qua users.json. Dùng --include-users nếu thật sự cần đồng bộ tài khoản.")
        target_files = [f for f in target_files if f != "users.json"]

    for hf_name in target_files:
        local_rel = sync_files.get(hf_name)
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
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
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
            bak_path = backup_local_file(local_path)
            if bak_path:
                print(f"       Backup  : {bak_path}")

        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

        # Merge
        if hf_name == "users.json" and isinstance(remote_data, dict):
            result = merge_users_dict(local_path, remote_data)
            print(
                f"       Local   : {result['total']} tài khoản "
                f"(mới: +{result['added']}, metadata cập nhật: {result['updated']})"
            )
        elif isinstance(remote_data, list):
            key_fn = FILE_KEY_FN.get(hf_name)
            result = merge_json_list(local_path, remote_data, key_fn)
            print(f"       Local   : {result['total']} bản ghi (mới thêm: +{result['added']})")
        else:
            write_json(local_path, remote_data, indent=2)
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
    parser.add_argument("--include-users", action="store_true", help="Cho phép đồng bộ users.json sau khi đã backup và xác nhận rủi ro")
    args = parser.parse_args()

    # Đọc config
    env_token, env_dataset = load_hf_config()
    token      = args.token   or env_token
    dataset_id = args.dataset or env_dataset

    if not dataset_id:
        print("\n❌ Chưa có HF_DATASET_ID.")
        print("   Cấu hình biến môi trường hoặc truyền: python sync_from_hf.py --dataset <owner/dataset>")
        sys.exit(1)

    if not token:
        print("\n⚠️  Chưa có HF_TOKEN!")
        print("   Cách 1: Thêm HF_TOKEN=<token> vào file .streamlit/secrets.toml")
        print("   Cách 2: Chạy:  set HF_TOKEN=<token>  (Windows) rồi chạy lại script")
        print("   Cách 3: Truyền:  python sync_from_hf.py --token <token>\n")
        confirm = input("Vẫn thử tải (dataset public không cần token)? [y/N] ").strip().lower()
        if confirm != "y":
            sys.exit(0)

    sync_from_hf(
        token      = token,
        dataset_id = dataset_id,
        files      = args.file,
        dry_run    = args.dry_run,
        backup     = not args.no_backup,
        include_users = args.include_users,
    )
