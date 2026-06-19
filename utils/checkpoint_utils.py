"""Checkpoint phân tích video — resume Pass 2 sau deploy HF / crash."""
import gzip
import hashlib
import json
import os
import pickle
import subprocess
import threading
import time

try:
    from .checksum import verify_sha256_sidecar, write_sha256_sidecar
except ImportError:
    from checksum import verify_sha256_sidecar, write_sha256_sidecar

CHECKPOINT_VERSION = 1
CHECKPOINT_INTERVAL_PASS2 = 300

_ckpt_locks = {}
_ckpt_locks_guard = threading.Lock()


def _ckpt_lock(path):
    with _ckpt_locks_guard:
        if path not in _ckpt_locks:
            _ckpt_locks[path] = threading.Lock()
        return _ckpt_locks[path]


class _LmPoint:
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z, visibility):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)


class _LmList:
    """Thay thế MediaPipe pose_landmarks khi nạp từ checkpoint."""

    def __init__(self, points):
        self.landmark = [_LmPoint(*p) for p in points] if points else []


def get_checkpoint_path(video_path, processed_dir):
    if not video_path:
        return ""
    clean = video_path.replace("\\", "/")
    h = hashlib.md5(clean.encode("utf-8")).hexdigest()
    return os.path.join(processed_dir, f"checkpoint_{h}.pkl.gz")


def build_config_hash(video_path, model_type, min_confidence, exercise_name, skip_step, resize_width):
    # KHÔNG dùng mtime — HF Space re-download video đổi mtime → checkpoint bị bỏ qua sai
    raw = "|".join(
        str(x)
        for x in (
            video_path or "",
            model_type or "",
            min_confidence,
            exercise_name or "",
            skip_step,
            resize_width,
        )
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def serialize_landmarks(lm):
    if lm is None:
        return None
    try:
        return [[p.x, p.y, p.z, p.visibility] for p in lm.landmark]
    except Exception:
        return None


def deserialize_landmarks(data):
    if data is None:
        return None
    return _LmList(data)


def serialize_pass1_item(item):
    return {
        "frame_idx": item.get("frame_idx"),
        "processed_count": item.get("processed_count"),
        "landmarks": serialize_landmarks(item.get("landmarks")),
        "detected": item.get("detected", False),
        "filtered_stranger": item.get("filtered_stranger", False),
        "goc_vai_left": item.get("goc_vai_left"),
        "goc_khuyu_left": item.get("goc_khuyu_left"),
        "goc_vai_right": item.get("goc_vai_right"),
        "goc_khuyu_right": item.get("goc_khuyu_right"),
        "goc_vai": item.get("goc_vai"),
        "goc_khuyu": item.get("goc_khuyu"),
    }


def deserialize_pass1_item(item):
    return {
        "frame_idx": item.get("frame_idx"),
        "processed_count": item.get("processed_count"),
        "landmarks": deserialize_landmarks(item.get("landmarks")),
        "detected": item.get("detected", False),
        "filtered_stranger": item.get("filtered_stranger", False),
        "goc_vai_left": item.get("goc_vai_left"),
        "goc_khuyu_left": item.get("goc_khuyu_left"),
        "goc_vai_right": item.get("goc_vai_right"),
        "goc_khuyu_right": item.get("goc_khuyu_right"),
        "goc_vai": item.get("goc_vai"),
        "goc_khuyu": item.get("goc_khuyu"),
    }


def save_checkpoint(path, data):
    if not path:
        return False
    lock = _ckpt_lock(path)
    tmp = None
    with lock:
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # hf_hub_download đôi khi tạo thư mục tại chính path khi download thất bại.
            # Phát hiện và dọn sạch trước khi ghi, tránh [Errno 21] IsADirectoryError.
            if os.path.isdir(path):
                import shutil as _shutil
                _shutil.rmtree(path, ignore_errors=True)
                print(f"[Checkpoint] Da xoa thu muc lam cang duong checkpoint: {path}")
            payload = dict(data)
            payload["version"] = CHECKPOINT_VERSION
            payload["saved_at"] = time.time()
            tmp = f"{path}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
            with gzip.open(tmp, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            # Xác minh file gzip đọc được trước khi thay thế — tránh file dở dang
            with gzip.open(tmp, "rb") as f:
                verified = pickle.load(f)
            if not isinstance(verified, dict):
                raise ValueError("checkpoint payload invalid after write")
            os.replace(tmp, path)
            write_sha256_sidecar(path)
            tmp = None
            return True
        except Exception as e:
            print(f"[Checkpoint] Loi ghi checkpoint: {e}")
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            return False


def load_checkpoint(path, retries=4, retry_delay=0.2):
    if not path or not os.path.exists(path):
        return None
    # Dọn sạch nếu path là thư mục (do hf_hub_download tạo khi download thất bại)
    if os.path.isdir(path):
        try:
            import shutil as _shutil
            _shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass
        return None
    if not verify_sha256_sidecar(path, required=True):
        print(f"[Checkpoint] Checksum checkpoint khong hop le hoac thieu: {os.path.basename(path)}")
        return None
    lock = _ckpt_lock(path)
    last_err = None
    for attempt in range(max(1, retries)):
        with lock:
            try:
                if os.path.getsize(path) < 64:
                    return None
                with gzip.open(path, "rb") as f:
                    data = pickle.load(f)
                if not isinstance(data, dict) or data.get("version") != CHECKPOINT_VERSION:
                    return None
                return data
            except Exception as e:
                last_err = e
        if attempt < retries - 1:
            time.sleep(retry_delay * (attempt + 1))
    err_s = str(last_err or "")
    if any(x in err_s for x in ("end-of-stream", "EOF", "truncated", "Compressed file ended")):
        try:
            bad = f"{path}.bad.{int(time.time())}"
            os.replace(path, bad)
            print(f"[Checkpoint] File checkpoint hỏng đã đổi tên -> {os.path.basename(bad)}")
        except Exception:
            pass
    else:
        print(f"[Checkpoint] Loi doc checkpoint: {last_err}")
    return None


def clear_checkpoint(path):
    if not path or not os.path.exists(path):
        return
    try:
        if os.path.isdir(path):
            # hf_hub_download tao THU MUC tai path khi download that bai -> os.remove se
            # bao [Errno 21] Is a directory. Dung rmtree (giong save/load_checkpoint).
            import shutil as _shutil
            _shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    except Exception as e:
        print(f"[Checkpoint] Loi xoa checkpoint: {e}")


def checkpoint_ui_progress(ckpt):
    """Ánh xạ checkpoint → % UI (cùng công thức bg_progress_callback)."""
    if not ckpt:
        return 0.01, "🚀 Đang chuẩn bị phân tích..."
    phase = ckpt.get("phase")
    if phase == "pass1_done":
        return 0.45, "🔄 Tiếp tục từ Bước 2 (Bước 1 đã lưu checkpoint)..."
    if phase == "pass2":
        done = int(ckpt.get("pass2_processed_count") or 0)
        total = len(ckpt.get("pass1_data") or [])
        if total > 0:
            internal_p = 0.5 + min(done / total, 1.0) * 0.40
            p2_pct = ((internal_p - 0.5) / 0.42) * 100
            prog_val = 0.50 + ((internal_p - 0.5) / 0.42) * 0.40
            return min(prog_val, 0.89), f"🔄 Tiếp tục Bước 2/2 từ frame {done}/{total} ({p2_pct:.0f}%)"
        return 0.50, "🔄 Tiếp tục Bước 2/2..."
    return 0.01, "🚀 Đang chuẩn bị phân tích..."


def assemble_video_from_jpgs(local_temp_dir, out_path, fps_export):
    """Ghép ảnh JPG đã ghi (resume Pass 2) thành MP4 bằng ffmpeg."""
    if not local_temp_dir or not os.path.isdir(local_temp_dir):
        return False
    jpgs = sorted(f for f in os.listdir(local_temp_dir) if f.lower().endswith(".jpg"))
    if not jpgs:
        return False
    pattern = os.path.join(local_temp_dir, "f_%06d.jpg")
    if not os.path.exists(os.path.join(local_temp_dir, "f_000001.jpg")):
        first = jpgs[0]
        if first != "f_000001.jpg":
            print(f"[Checkpoint] Canh bao: khong co f_000001.jpg, dung {len(jpgs)} anh")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(max(10, int(fps_export))),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-crf", "28",
        out_path,
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=3600)
        ok = r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 5 * 1024
        if not ok:
            print(f"[Checkpoint] ffmpeg ghep video that bai: {r.stderr[-500:] if r.stderr else ''}")
        return ok
    except Exception as e:
        print(f"[Checkpoint] Loi ghep video tu JPG: {e}")
        return False
