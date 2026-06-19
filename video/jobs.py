"""Background analysis job registry helpers.

This module owns the in-memory thread/cancel/slot registry. The heavy analysis
body is still provided by app-level services so importing this module never
starts work or pulls in Streamlit.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


ProgressLoader = Callable[[str], dict[str, Any] | None]


class AnalysisSlotPool:
    """Manage concurrent analysis slots and release stale holders."""

    def __init__(
        self,
        max_slots: int,
        running_threads: dict[str, threading.Thread],
        load_progress_fn: ProgressLoader | None = None,
        orphan_seconds: int = 90,
    ) -> None:
        self.max_slots = max(1, max_slots)
        self._running_threads = running_threads
        self._load_progress_fn = load_progress_fn
        self._orphan_seconds = orphan_seconds
        self._lock = threading.Lock()
        self._holders: dict[str, float] = {}

    def configure(
        self,
        *,
        load_progress_fn: ProgressLoader | None = None,
        orphan_seconds: int | None = None,
    ) -> None:
        if load_progress_fn is not None:
            self._load_progress_fn = load_progress_fn
        if orphan_seconds is not None:
            self._orphan_seconds = orphan_seconds

    def _load_progress(self, video_path: str) -> dict[str, Any] | None:
        if self._load_progress_fn is None:
            return None
        try:
            return self._load_progress_fn(video_path)
        except Exception:
            return None

    def _purge_dead(self) -> None:
        for vp in list(self._holders.keys()):
            t = self._running_threads.get(vp)
            if t is None or not t.is_alive():
                self._holders.pop(vp, None)
                continue
            prog = self._load_progress(vp)
            if not prog or prog.get("status") != "processing":
                self._holders.pop(vp, None)
                continue
            hb = float(prog.get("heartbeat") or prog.get("start_time") or 0)
            if hb and (time.time() - hb) > self._orphan_seconds:
                self._holders.pop(vp, None)
                self._running_threads.pop(vp, None)

    def try_acquire(self, video_path: str, priority: bool = False, timeout: float = 2.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                self._purge_dead()
                if priority and len(self._holders) >= self.max_slots:
                    oldest_vp, oldest_hb = None, float("inf")
                    for vp in self._holders:
                        prog = self._load_progress(vp) or {}
                        hb = float(prog.get("heartbeat") or prog.get("start_time") or 0) or time.time()
                        if hb < oldest_hb:
                            oldest_hb, oldest_vp = hb, vp
                    if oldest_vp and (time.time() - oldest_hb) > 45:
                        self._holders.pop(oldest_vp, None)
                if video_path in self._holders or len(self._holders) < self.max_slots:
                    self._holders[video_path] = time.time()
                    return True
            time.sleep(0.12)
        return False

    def release(self, video_path: str) -> None:
        with self._lock:
            self._holders.pop(video_path, None)

    def holder_summary(self) -> list[str]:
        with self._lock:
            self._purge_dead()
            names = []
            for vp in self._holders:
                prog = self._load_progress(vp) or {}
                names.append(prog.get("video_name") or os.path.basename(vp))
            return names


@dataclass
class AnalysisRequest:
    username: str
    video_path: str
    video_name: str
    exercise_name: str
    phase_key: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisJobRegistry:
    max_concurrent: int
    load_progress_fn: ProgressLoader | None = None
    orphan_seconds: int = 90
    running_threads: dict[str, threading.Thread] = field(default_factory=dict)
    cancel_flags: dict[str, threading.Event] = field(default_factory=dict)
    start_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.slots = AnalysisSlotPool(
            self.max_concurrent,
            self.running_threads,
            load_progress_fn=self.load_progress_fn,
            orphan_seconds=self.orphan_seconds,
        )

    def configure_progress_loader(
        self,
        load_progress_fn: ProgressLoader,
        *,
        orphan_seconds: int | None = None,
    ) -> None:
        self.load_progress_fn = load_progress_fn
        if orphan_seconds is not None:
            self.orphan_seconds = orphan_seconds
        self.slots.configure(
            load_progress_fn=load_progress_fn,
            orphan_seconds=orphan_seconds,
        )

    def cleanup_dead_thread(self, video_path: str) -> None:
        if not video_path:
            return
        t = self.running_threads.get(video_path)
        if t and not t.is_alive():
            self.running_threads.pop(video_path, None)

    def is_thread_running(self, video_path: str | None = None) -> bool:
        if video_path:
            t = self.running_threads.get(video_path)
            return bool(t and t.is_alive())
        return any(t.is_alive() for t in self.running_threads.values() if t)


def start_analysis_job(
    *,
    registry: AnalysisJobRegistry,
    video_path: str,
    video_name: str | None,
    target: Callable[[], None],
    force_restart: bool = False,
) -> dict[str, Any]:
    """Atomically register and start one background analysis thread."""

    with registry.start_lock:
        current = registry.running_threads.get(video_path)
        if current is not None and current.is_alive():
            if not force_restart:
                print(f"[BG Process] Luong khac vua khoi chay '{video_name}' - bo luong trung nay.")
                return {"started": False, "reason": "already_running"}
            print("[BG Process] force_restart=True - ghi de luong cu.")

        old_flag = registry.cancel_flags.get(video_path)
        if old_flag:
            old_flag.set()
        registry.cancel_flags[video_path] = threading.Event()

        thread = threading.Thread(target=target, daemon=True)
        registry.running_threads[video_path] = thread
        thread.start()
        return {"started": True, "reason": ""}


def _bind_deps(deps: Any) -> None:
    if deps is None:
        return
    globals().update(
        {k: v for k, v in vars(deps).items() if not k.startswith("__")}
    )


def start_background_analysis(
    deps,
    video_path,
    username,
    full_name,
    video_name,
    exercise_name,
    giai_doan,
    model_type,
    confidence,
    temp_uploaded_path=None,
    skip_step=None,
    resize_width=None,
    force_train_classifier=False,
    force_restart=False,
):
    """Khởi chạy tiến trình phân tích video dưới background thread.
    Trả về dict: started (bool), reason (str)."""
    _bind_deps(deps)

    if not video_path:
        return {"started": False, "reason": "no_video"}

    # CHONG CHAY TRUNG (som, khong side-effect): da co luong song dang phan tich video nay
    # va khong force_restart -> bo qua NGAY, tranh setup thua va tranh dung cancel-flag cua
    # luong dang chay. Kiem tra atomic lan cuoi o cuoi ham (truoc khi start).
    _ht = _running_threads.get(video_path)
    if (not force_restart) and _ht is not None and _ht.is_alive():
        print(f"[BG Process] Da co luong dang phan tich '{video_name or video_path}' — bo qua khoi chay trung.")
        return {"started": False, "reason": "already_running"}

    _don_dep_thread_phan_tich(video_path)

    ckpt_path = get_checkpoint_path(video_path, PROCESSED_DIR)
    ckpt_existing = load_checkpoint(ckpt_path)
    has_ckpt = (
        not force_restart
        and bool(
            ckpt_existing
            and ckpt_existing.get("pass1_data")
            and ckpt_existing.get("phase") in ("pass1_done", "pass2")
        )
    )

    if has_ckpt:
        model_type = ckpt_existing.get("model_type") or model_type
        skip_step = ckpt_existing.get("skip_step") if ckpt_existing.get("skip_step") is not None else skip_step
        resize_width = ckpt_existing.get("resize_width") or resize_width
        force_train_classifier = False
    else:
        model_type, skip_step, resize_width = tinh_tham_so_toc_do_phan_tich(
            video_path, exercise_name, model_type, skip_step, resize_width
        )
    skip_step = skip_step_theo_model(model_type, skip_step)

    # (Cancel-flag duoc tao/đặt o BLOCK ATOMIC cuoi ham — chi khi CHAC CHAN start luong moi.
    #  Truoc day set flag o day la SAI: lo set flag cua luong dang chay roi van bo qua khoi
    #  chay -> giet luong cu ma khong start luong moi.)

    if has_ckpt:
        cfg_now = build_config_hash(video_path, model_type, confidence, exercise_name, skip_step, resize_width)
        if ckpt_existing.get("config_hash") != cfg_now:
            print(f"[Checkpoint] Hash khong khop — xoa checkpoint cu va chay lai tu dau")
            clear_checkpoint(ckpt_path)
            has_ckpt = False

    # (Kiem tra trung + xu ly force_restart da chuyen xuong block atomic o cuoi ham.)

    job_meta = {
        "full_name": full_name,
        "exercise_name": exercise_name,
        "giai_doan": giai_doan,
        "model_type": model_type,
        "confidence": confidence,
        "skip_step": skip_step,
        "resize_width": resize_width,
        "force_train_classifier": force_train_classifier,
    }
    snap = _tien_do_phan_tich_hien_tai(video_path, ckpt_existing if has_ckpt else None)
    write_progress(
        video_path, "processing", username=username, video_name=video_name,
        progress=snap["progress"], elapsed=snap["elapsed"],
        start_time=snap["start_time"],
        status_msg=snap["status_msg"], job_meta=job_meta,
    )

    def thread_target():
        nonlocal video_path
        # Hạ ưu tiên CPU (nice) của luồng phân tích — và các thread con MediaPipe/
        # OpenCV mà nó tạo ra đều kế thừa mức nice này trên Linux. Nhờ đó scheduler
        # luôn ưu tiên luồng UI của Streamlit khi tranh chấp CPU → web KHÔNG bị
        # đơ/đứng khi đang phân tích, hoặc khi job tự chạy lại lúc tải trang.
        # Khi UI rảnh, phân tích vẫn dùng full CPU nên không làm chậm lúc nhàn rỗi.
        try:
            if hasattr(os, "nice"):
                os.nice(10)
        except Exception:
            pass
        progress_video_path = video_path
        start_t = snap["start_time"]
        sem_acquired = False
        # Capture flag của thread này — không lookup từ dict để tránh lấy nhầm flag của thread mới hơn
        _my_cancel_flag = _cancel_flags.get(video_path)
        ckpt_wait = load_checkpoint(get_checkpoint_path(progress_video_path, PROCESSED_DIR))
        is_resume_pass2 = bool(
            ckpt_wait
            and ckpt_wait.get("pass1_data")
            and ckpt_wait.get("phase") in ("pass1_done", "pass2")
        )

        try:
            # HÀNG ĐỢI: tối đa MAX_CONCURRENT_ANALYSIS video — tự nhả slot job chết
            wait_started = time.time()
            while not sem_acquired:
                sem_acquired = _analysis_slots.try_acquire(
                    progress_video_path, priority=is_resume_pass2, timeout=2.0
                )
                if not sem_acquired:
                    waited = time.time() - wait_started
                    holders = _analysis_slots.holder_summary()
                    holder_txt = ", ".join(holders[:2]) if holders else "video khác"
                    if is_resume_pass2:
                        q_prog = 0.45
                        step = "Bước 2 sẵn sàng — đang chờ slot CPU"
                    else:
                        q_prog = min(0.44, max(0.02, waited / 600.0 * 0.44))
                        step = "Đang chờ slot phân tích"
                    write_progress(
                        progress_video_path, "processing", username=username, video_name=video_name,
                        progress=q_prog, elapsed=time.time() - start_t, start_time=start_t,
                        status_msg=(
                            f"⏳ {step} ({waited:.0f}s) — đang chạy: **{holder_txt}** "
                            f"(tối đa {MAX_CONCURRENT_ANALYSIS} video/lúc)"
                        ),
                    )
                    if waited > 900:
                        _analysis_slots._purge_dead()

            boot_snap = _tien_do_phan_tich_hien_tai(progress_video_path)
            write_progress(
                progress_video_path, "processing", username=username, video_name=video_name,
                progress=max(boot_snap["progress"], 0.02), elapsed=time.time() - start_t,
                start_time=start_t,
                status_msg=boot_snap["status_msg"] or "🚀 Đang khởi tạo luồng phân tích...",
            )
            # Bước A: Nếu có tệp tải lên tạm thời, thực hiện nén/FFmpeg trong background trước
            if temp_uploaded_path:
                write_progress(progress_video_path, "processing", username=username, video_name=video_name, progress=0.05, elapsed=0.0, start_time=start_t, status_msg="⚙️ Đang tối ưu hóa định dạng video (H.264)...")
                try:
                    v_codec = None
                    try: v_codec, _ = get_video_codec(temp_uploaded_path)
                    except: pass

                    is_h264_mp4 = (v_codec == 'h264' and os.path.splitext(temp_uploaded_path)[1].lower() == '.mp4')
                    if is_h264_mp4:
                        if os.path.exists(video_path):
                            try: os.remove(video_path)
                            except: pass
                        os.rename(temp_uploaded_path, video_path)
                    else:
                        # Đổi mã hóa sang H.264
                        video_path_mp4 = video_path.rsplit('.', 1)[0] + ".mp4"
                        cmd = build_background_upload_h264_command(
                            temp_uploaded_path,
                            video_path_mp4,
                            ffmpeg_threads=MAX_FFMPEG_THREADS,
                        )
                        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        start_ffmpeg_time = time.time()
                        while process.poll() is None:
                            time.sleep(1.0)
                            elapsed_ffmpeg = time.time() - start_ffmpeg_time
                            # mock_prog from 0.05 to 0.12 during first 60 seconds
                            mock_prog = 0.05 + min(elapsed_ffmpeg / 60.0, 1.0) * 0.07
                            write_progress(
                                progress_video_path, "processing",
                                username=username, video_name=video_name,
                                progress=mock_prog, elapsed=time.time() - start_t,
                                start_time=start_t,
                                status_msg=f"⚙️ Đang tối ưu hóa định dạng video (H.264)... ({elapsed_ffmpeg:.0f}s)"
                            )

                        is_compress_ok = False
                        if process.returncode == 0 and os.path.exists(video_path_mp4) and os.path.getsize(video_path_mp4) > 5 * 1024:
                            try:
                                mtime_c = os.path.getmtime(video_path_mp4)
                                size_c = os.path.getsize(video_path_mp4)
                                is_compress_ok = _check_video_valid_cached(video_path_mp4, mtime_c, size_c)
                            except: pass

                        if is_compress_ok:
                            try: os.remove(temp_uploaded_path)
                            except: pass
                            video_path = video_path_mp4
                        else:
                            if os.path.exists(video_path_mp4):
                                try: os.remove(video_path_mp4)
                                except: pass
                            if os.path.exists(video_path):
                                try: os.remove(video_path)
                                except: pass
                            os.rename(temp_uploaded_path, video_path)
                except Exception as compress_err:
                    print(f"[NCV Compress BG] Lỗi nén: {compress_err}")
                    if os.path.exists(temp_uploaded_path):
                        if os.path.exists(video_path):
                            try: os.remove(video_path)
                            except: pass
                        os.rename(temp_uploaded_path, video_path)

            write_progress(progress_video_path, "processing", username=username, video_name=video_name, progress=0.10, elapsed=time.time()-start_t, start_time=start_t, status_msg="⬇️ Đang kiểm tra video cục bộ...")

            # Tối ưu hóa: Nếu video gốc không có sẵn local và chưa có H264 (_f.mp4) local,
            # kiểm tra xem đã có _f.mp4 trên cloud chưa để tải và phân tích trực tiếp cho nhanh!
            analysis_input_path = video_path
            final_h264 = get_final_h264_path(video_path)
            orig_video_path = _strip_to_original_upload(video_path)  # stripped path (không có _f)
            is_raw_local = os.path.exists(video_path) and os.path.getsize(video_path) >= 5 * 1024

            # Khi video_path là _f.mp4 mà chưa có local → thử file gốc (user vừa upload, _f chưa tạo)
            if not is_raw_local and orig_video_path and orig_video_path != video_path:
                if os.path.exists(orig_video_path) and os.path.getsize(orig_video_path) >= 5 * 1024:
                    analysis_input_path = orig_video_path
                    is_raw_local = True
                    write_progress(progress_video_path, "processing", username=username, video_name=video_name,
                                   progress=0.10, elapsed=time.time()-start_t, start_time=start_t,
                                   status_msg="✅ Sử dụng video gốc BN để phân tích (H.264 sẽ được tạo tự động)...")

            if not is_raw_local:
                if final_h264 != video_path:
                    write_progress(progress_video_path, "processing", username=username, video_name=video_name, progress=0.10, elapsed=time.time()-start_t, start_time=start_t, status_msg="⬇️ Kiểm tra video H.264 đã tối ưu...")
                    dl_h264_ok = download_file_with_progress(final_h264, write_progress, start_t, username, video_name)
                    if dl_h264_ok:
                        analysis_input_path = final_h264
                        print(f"[BG Process] Chuyển đổi sang phân tích H264 đã tối ưu: {final_h264}")

            if analysis_input_path == video_path and not is_raw_local:
                write_progress(progress_video_path, "processing", username=username, video_name=video_name, progress=0.12, elapsed=time.time()-start_t, start_time=start_t, status_msg="⬇️ Đang tải video gốc từ Cloud về server...")
                try:
                    dl_ok = download_file_with_progress(video_path, write_progress, start_t, username, video_name)
                    if dl_ok:
                        write_progress(progress_video_path, "processing", username=username, video_name=video_name, progress=0.18, elapsed=time.time()-start_t, start_time=start_t, status_msg="✅ Đã tải video xong, đang chuẩn bị phân tích...")
                    else:
                        # Fallback 1: thử tải file gốc (stripped _f) nếu có trên HF Dataset
                        if orig_video_path and orig_video_path != video_path:
                            dl_orig_ok = download_file_with_progress(orig_video_path, write_progress, start_t, username, video_name)
                            if dl_orig_ok:
                                analysis_input_path = orig_video_path
                                dl_ok = True
                                write_progress(progress_video_path, "processing", username=username, video_name=video_name, progress=0.18, elapsed=time.time()-start_t, start_time=start_t, status_msg="✅ Đã tải video gốc BN, đang chuẩn bị phân tích...")

                        # Fallback 2: thử tải _f.mp4 (nếu khác video_path)
                        if not dl_ok and final_h264 != video_path:
                            dl_h264_ok = download_file_with_progress(final_h264, write_progress, start_t, username, video_name)
                            if dl_h264_ok:
                                analysis_input_path = final_h264
                                dl_ok = True
                                write_progress(progress_video_path, "processing", username=username, video_name=video_name, progress=0.18, elapsed=time.time()-start_t, start_time=start_t, status_msg="✅ Đã tải video H.264 tối ưu, đang chuẩn bị phân tích...")

                        if not dl_ok:
                            write_progress(progress_video_path, "error", username=username, video_name=video_name, progress=0.0, elapsed=time.time()-start_t, start_time=start_t, error_msg="❌ Video gốc không còn trên server và không có trên Cloud — vui lòng **tải lên lại video** để phân tích.")
                            return
                except Exception as dl_err:
                    print(f"[BG Download] Lỗi tải video: {dl_err}")
                    write_progress(progress_video_path, "error", username=username, video_name=video_name, progress=0.0, elapsed=time.time()-start_t, start_time=start_t, error_msg=f"❌ Lỗi tải video: {dl_err}")
                    return
            else:
                if analysis_input_path == final_h264:
                    write_progress(progress_video_path, "processing", username=username, video_name=video_name, progress=0.18, elapsed=time.time()-start_t, start_time=start_t, status_msg="✅ Sử dụng H.264 đã tối ưu, đang khởi động AI...")
                else:
                    write_progress(progress_video_path, "processing", username=username, video_name=video_name, progress=0.18, elapsed=time.time()-start_t, start_time=start_t, status_msg="✅ Video đã có sẵn, đang khởi động AI...")

            # Bước B: Nạp cấu hình bài tập chuẩn
            ex_key = next((k for k in BAI_TAP if BAI_TAP[k]['ten'] == exercise_name), 'codman')
            bt = BAI_TAP[ex_key]

            ss_override = PHASE_ERROR_DEFAULT
            if "Giai đoạn 1" in giai_doan:
                ss_override = PHASE_ERROR["g1"]
            elif "Giai đoạn 3" in giai_doan:
                ss_override = PHASE_ERROR["g3"]

            bt_chuan_ncv = bt['chuan'].copy()
            bt_chuan_ncv['sai_so'] = ss_override

            bt_ncv = bt.copy()
            bt_ncv['chuan'] = bt_chuan_ncv

            # Callback cập nhật tiến độ cho MediaPipe
            last_write_time = [0.0]
            last_prog_tenth = [-1]

            def bg_progress_callback(p, frame_count=None, total_frames=None):
                now = time.time()
                elap = now - start_t
                # Chia tiến độ thành các vùng để video lớn không bị đứng ở 99/100% quá lâu:
                # tải/chuẩn bị 0-18%, Pass 1 18-45%, train/cấu hình 45-50%, Pass 2 50-90%,
                # ghi ảnh/zip/đóng gói 90-99%.
                if p <= 0.5:
                    prog_val = 0.18 + (p / 0.5) * 0.27
                elif p <= 0.505:
                    prog_val = 0.45
                elif p <= 0.92:
                    prog_val = 0.50 + ((p - 0.5) / 0.42) * 0.40
                else:
                    prog_val = min(p, 0.99)

                # Tạo status_msg sinh động để hiển thị chi tiết tiến trình
                _fc = f" | Frame {frame_count}/{total_frames}" if (frame_count and total_frames) else ""
                if p <= 0.5:
                    p1_pct = (p / 0.5) * 100
                    status_msg = f"🔬 Bước 1/2: Trích xuất khung xương ({p1_pct:.0f}%){_fc}"
                elif p <= 0.505:
                    status_msg = "🤖 Đang chuẩn bị model ML và khởi động Pass 2..."
                elif p < 0.92:
                    p2_pct = ((p - 0.5) / 0.42) * 100
                    status_msg = f"🎨 Bước 2/2: Vẽ nhãn REF/ML & ghi khung hình ({p2_pct:.0f}%){_fc}"
                else:
                    status_msg = "📦 Đang lưu frames, đóng gói video và hoàn tất kết quả..."

                percent_tenth = int(prog_val * 1000)
                # Ghi tiến độ mỗi 1s (heartbeat) hoặc khi % thay đổi — UI không bị đứng im.
                if percent_tenth != last_prog_tenth[0] or (now - last_write_time[0] >= 1.0):
                    # Kiểm tra cancel flag riêng của thread này (không lookup dict tránh lấy nhầm flag mới)
                    if _my_cancel_flag and _my_cancel_flag.is_set():
                        raise InterruptedError("⛔ Phân tích bị dừng bởi người dùng.")
                    write_progress(progress_video_path, "processing", username=username, video_name=video_name, progress=prog_val, elapsed=elap, start_time=start_t, status_msg=status_msg)
                    last_write_time[0] = now
                    last_prog_tenth[0] = percent_tenth

            # Bước C: Xác thực file video mở được bằng OpenCV (tránh "Video Error" do file tạm/hỏng)
            resolved_analysis = _dam_bao_video_cho_phan_tich(
                analysis_input_path, username=username, video_name=video_name
            )
            if not resolved_analysis:
                write_progress(
                    progress_video_path, "error", username=username, video_name=video_name,
                    progress=0.0, elapsed=time.time() - start_t, start_time=start_t,
                    error_msg=(
                        "Không mở được video để phân tích — file upload có thể hỏng hoặc chưa có trên Cloud. "
                        "Thử F5, hoặc upload lại video gốc."
                    ),
                )
                return
            analysis_input_path = resolved_analysis
            # Báo hiệu đang khởi tạo MediaPipe + thông tin tối ưu tốc độ
            _skip_info = f", bỏ {skip_step} frame" if skip_step and int(skip_step) > 0 else ""
            _res_info = f", {resize_width}p" if resize_width else ""
            write_progress(
                progress_video_path, "processing", username=username, video_name=video_name,
                progress=0.19, elapsed=time.time() - start_t, start_time=start_t,
                status_msg=f"🤖 Đang khởi tạo AI (MediaPipe{_res_info}{_skip_info})... vui lòng đợi ~30s",
            )

            # Bước D: Chạy phân tích AI trích xuất xương
            output_path, ref_name_detected, _, angle_data, total_frames, valid_frames, temp_folder, zip_data, frame_paths, _, all_frames_data, all_warnings = xu_ly_video_day_du(
                analysis_input_path, bt_chuan_ncv, bg_progress_callback,
                model_type=model_type, min_confidence=confidence,
                exercise_name=exercise_name,
                skip_step=skip_step, resize_width=resize_width,
                force_train_classifier=force_train_classifier,
                checkpoint_video_path=progress_video_path,
            )

            elap = time.time() - start_t
            # Heartbeat ngay sau xu_ly_video_day_du: post-processing có thể mất vài phút
            # (tính metrics, CSV, HF upload) mà không gọi callback → heartbeat cũ → UI đứng im
            write_progress(progress_video_path, "processing", username=username, video_name=video_name,
                           progress=0.92, elapsed=elap, start_time=start_t,
                           status_msg="📊 Đang tính toán metrics và lưu kết quả...")

            if valid_frames > 0 and len(angle_data) > 0:
                df = pd.DataFrame(angle_data)
                metrics = tinh_metrics_chi_tiet(df, bt_ncv)
                phase_bounds_for_ml = None

                is_gay_ex = any(kw in str(exercise_name or '').lower() for kw in ["gậy", "gay", "pulley", "stick"])
                if is_gay_ex:
                    metrics_overall = recalc_metrics(df, ss_override, bt_ncv.get('ten', ''))
                    metrics_g1 = metrics_overall
                    metrics_g2 = metrics_overall
                    metrics_g3 = metrics_overall
                    metrics["ty_le_tong_the"] = metrics_overall["do_chinh_xac"]
                else:
                    bounds = segment_frames(df)
                    n0, n1, n2, n3 = bounds
                    phase_bounds_for_ml = bounds
                    df_g1 = df.iloc[n0:n1]
                    df_g2 = df.iloc[n1:n2]
                    df_g3 = df.iloc[n2:n3]
                    metrics_g1 = recalc_metrics(df_g1, PHASE_ERROR["g1"], bt_ncv.get('ten', ''))
                    metrics_g2 = recalc_metrics(df_g2, PHASE_ERROR["g2"], bt_ncv.get('ten', ''))
                    metrics_g3 = recalc_metrics(df_g3, PHASE_ERROR["g3"], bt_ncv.get('ten', ''))

                stats_data = {
                    "do_chinh_xac": metrics["ty_le_tong_the"],
                    "ty_le_gan_dung": metrics["ty_le_gan_dung"],
                    "ty_le_vai_dung": metrics["ty_le_vai_dung"],
                    "ty_le_khuyu_dung": metrics["ty_le_khuyu_dung"],
                    "frame_dung": metrics["frame_dung"],
                    "frame_gan_dung": metrics["frame_gan_dung"],
                    "tong_frame_hop_le": valid_frames,
                    "tb_goc_vai": metrics["tb_goc_vai"],
                    "tb_goc_khuyu": metrics["tb_goc_khuyu"],
                    "min_goc_vai": metrics["min_goc_vai"],
                    "max_goc_vai": metrics["max_goc_vai"],
                    "min_goc_khuyu": metrics["min_goc_khuyu"],
                    "max_goc_khuyu": metrics["max_goc_khuyu"],
                    "std_goc_vai": metrics["std_goc_vai"],
                    "std_goc_khuyu": metrics["std_goc_khuyu"],
                    "mae_tong": metrics["mae_tong"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "icc": metrics["icc"],
                    "tb_vai_chuan": metrics.get("tb_vai_chuan", 90),
                    "tb_khuyu_chuan": metrics.get("tb_khuyu_chuan", 170),
                    "thoi_gian": elap,
                    "tong_frame": total_frames,
                    "warnings": all_warnings,
                    "metrics_g1": metrics_g1,
                    "metrics_g2": metrics_g2,
                    "metrics_g3": metrics_g3
                }

                # Lưu DataFrame ra CSV và giải phóng RAM
                if apply_classifier_to_dataframe and get_pose_classifier_status:
                    try:
                        if get_pose_classifier_status(DB_DIR).get("ready"):
                            df, ml_result = apply_classifier_to_dataframe(
                                df,
                                db_dir=DB_DIR,
                                phase_bounds=phase_bounds_for_ml,
                                exercise_name=exercise_name
                            )
                            stats_data = merge_ml_metrics(stats_data, ml_result)
                    except FileNotFoundError:
                        pass
                    except Exception as ml_err:
                        print(f"[Pose Classifier] Bo qua du doan ML cho video hien tai: {ml_err}")

                write_progress(progress_video_path, "processing", username=username, video_name=video_name,
                               progress=0.95, elapsed=time.time()-start_t, start_time=start_t,
                               status_msg="💾 Đang ghi CSV và lưu kết quả...")
                df_csv_path = output_path.replace('.mp4', '_data.csv')
                df.to_csv(df_csv_path, index=False)

                # Cập nhật loại bài tập chuẩn
                correct_ex_name = "Bài tập con lắc Codman"
                if ref_name_detected == "gay":
                    correct_ex_name = "Bài tập với gậy (Pulley Exercise)"
                elif ref_name_detected == "day":
                    correct_ex_name = "Bài tập với dây kháng lực (Theraband)"

                # Hàm cập nhật video_list.json an toàn đa luồng
                def cap_nhat_ds_video(video_list):
                    found_vid = None
                    for x_v in video_list:
                        if x_v.get('username') == username and (x_v.get('video_path') == video_path or x_v.get('video_name') == video_name):
                            found_vid = x_v
                            break

                    new_vid_record = {
                        "username": username,
                        "full_name": full_name,
                        "video_name": video_name,
                        "exercise": correct_ex_name,
                        "accuracy": round(metrics["ty_le_tong_the"], 1),
                        "time": get_vn_now().strftime("%H:%M - %d/%m/%Y"),
                        "video_path": video_path,
                        "processed_path": output_path,
                        "metrics": stats_data,
                        "df_path": df_csv_path,
                        "all_frames_data_path": all_frames_data,
                        "frames_zip": zip_data,
                        "frames_zip_path": zip_data,
                        "status": "Đã phân tích",
                        "sai_so": ss_override,
                        "giai_doan": giai_doan
                    }

                    if found_vid:
                        found_vid.update(new_vid_record)
                    else:
                        video_list.append(new_vid_record)
                    return video_list

                doc_lock_save_data(VIDEOS_FILE, cap_nhat_ds_video)

                # Đồng bộ tên exercise trong các JSON khác
                try:
                    dong_bo_va_chuan_hoa_exercise(
                        username=username,
                        video_name=video_name,
                        video_path=video_path,
                        original_exercise=correct_ex_name
                    )
                except Exception as sync_err:
                    print(f"[BG Process] Lỗi đồng bộ exercise: {sync_err}")

                # Ghi lịch sử tập luyện an toàn đa luồng (theo BN + thời gian phân tích xong)
                history_file = HISTORY_FILE
                hoan_tat_luc = get_vn_now().strftime("%H:%M - %d/%m/%Y")
                new_entry = {
                    "ngay": hoan_tat_luc,
                    "username": username,
                    "full_name": full_name,
                    "video_name": video_name,
                    "bai_tap": bt['ten'],
                    "accuracy": round(metrics["ty_le_tong_the"], 1),
                    "f1": round(metrics["f1_score"], 2),
                    "thoi_gian_tap": round(elap, 1),
                }

                def cap_nhat_lich_su(history_data):
                    key = _lich_su_entry_key(new_entry)
                    for h in history_data:
                        if _lich_su_entry_key(h) == key:
                            h.update(new_entry)
                            return history_data
                    history_data.append(new_entry)
                    return history_data

                try:
                    doc_lock_save_data(history_file, cap_nhat_lich_su)
                except Exception as hist_err:
                    print(f"[BG Process] Lỗi lưu lịch sử: {hist_err}")

                # Đồng bộ file lên Hugging Face Dataset dưới dạng bất đồng bộ
                push_file_to_hf_async(df_csv_path)
                push_file_to_hf_async(output_path)
                push_file_to_hf_async(all_frames_data)
                if zip_data:
                    push_file_to_hf_async(zip_data)
                h264_out = resolve_playback_video_path(output_path)
                if h264_out and h264_out != output_path and os.path.exists(h264_out):
                    push_file_to_hf_async(h264_out)

                # Lưu kết quả hoàn tất vào progress file
                result_data = {
                    "stats": stats_data,
                    "processed_video_path": output_path,
                    "df_path": df_csv_path,
                    "all_frames_data_path": all_frames_data,
                    "exercise": bt_ncv,
                    "frames_zip": zip_data,
                    "frames_zip_path": zip_data,
                    "temp_frames_dir": temp_folder
                }
                write_progress(progress_video_path, "success", username=username, video_name=video_name, progress=1.0, elapsed=elap, start_time=start_t, result=result_data)
            else:
                write_progress(progress_video_path, "error", username=username, video_name=video_name, progress=1.0, elapsed=elap, start_time=start_t, error_msg="Không thể trích xuất khung xương từ video (0 frame hợp lệ).")
        except InterruptedError as e:
            # Người dùng bấm Dừng — progress file đã được set "error" bởi _dung_phan_tich()
            print(f"[BG Process] Phân tích bị dừng bởi người dùng: {e}")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[BG Process] Lỗi trong background thread: {e}\n{tb}")
            elap = time.time() - start_t
            write_progress(progress_video_path, "error", username=username, video_name=video_name, progress=1.0, elapsed=elap, start_time=start_t, error_msg=str(e))
        finally:
            if sem_acquired:
                _analysis_slots.release(progress_video_path)
            _cancel_flags.pop(progress_video_path, None)

    # KIEM TRA-VA-START ATOMIC: dam bao CHI 1 luong / 1 video du co nhieu lan goi dong thoi
    # (resume cold-start + watcher dinh ky + user bam). Chi den day moi dung vao cancel-flag.
    return start_analysis_job(
        registry=_analysis_registry,
        video_path=video_path,
        video_name=video_name,
        target=thread_target,
        force_restart=force_restart,
    )
