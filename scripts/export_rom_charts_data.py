#!/usr/bin/env python3
"""
Xuất số liệu BIÊN ĐỘ ROM + Boxplot khớp cho 8 video nghiên cứu.

Nguồn chính: database/video_list.json (cùng metrics hiển thị trên web).
Tùy chọn --hf / --csv: tải CSV từ HF Dataset để thống kê boxplot theo nhóm Đúng/Gần đúng/Sai.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..")
BASE = os.path.join(ROOT, "database")
OUT_TXT = os.path.join(ROOT, "docs", "_ROM_CHARTS_DATA.txt")
OUT_JSON = os.path.join(ROOT, "docs", "_ROM_CHARTS_DATA.json")
HF_DATASET_ID = "quynhphuong1209/Rehab-AI-Monitor-2026-data"

sys.path.insert(0, SCRIPT_DIR)
from export_report_metrics import lay_metrics_chinh, phase_block, r1  # noqa: E402

BN_NGHIEN_CUU = (
    "Cao Thị Thường",
    "Hoàng Hạnh Nguyên",
    "Nguyễn Thị Nga",
    "Vũ Thị Hòa",
)
BAI_NGHIEN_CUU = (
    "Bài tập con lắc Codman",
    "Bài tập với gậy (Pulley Exercise)",
)
PHASE_KEYS = [
    ("metrics_g1", "G1", "±45°"),
    ("metrics_g2", "G2", "±30°"),
    ("metrics_g3", "G3", "±15°"),
]
DISPLAY_CSV_COLS = [
    "frame", "goc_vai", "goc_khuyu", "dung", "gan_dung",
]
BOX_GROUPS = ("ĐÚNG (Pass)", "GẦN ĐÚNG (Nearly)", "SAI (Fail)")


def parse_t(s):
    if not s:
        return datetime.min
    for fmt in ("%H:%M - %d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(s).strip(), fmt)
        except Exception:
            pass
    return datetime.min


def slot_key(username, exercise):
    return (str(username or "").strip(), str(exercise or "").strip())


def dedup_evals(evals):
    latest = {}
    for e in evals or []:
        k = (e.get("patient_username"), e.get("exercise"), e.get("doctor_username"))
        if k not in latest or parse_t(e.get("time")) >= parse_t(latest[k].get("time")):
            latest[k] = e
    return list(latest.values())


def has_doc_eval(evals, pu, ex):
    for e in evals:
        if e.get("patient_username") == pu and e.get("exercise") == ex and e.get("doctor_username") != "AI_Researcher":
            return True
    return False


def chon_video_moi_hon(a, b):
    ta = parse_t(a.get("time"))
    tb = parse_t(b.get("time"))
    if ta != tb:
        return a if ta >= tb else b
    acc_a = float(a.get("accuracy") or (a.get("metrics") or {}).get("do_chinh_xac") or 0)
    acc_b = float(b.get("accuracy") or (b.get("metrics") or {}).get("do_chinh_xac") or 0)
    return a if acc_a >= acc_b else b


def load_video_nghien_cuu():
    vl = json.load(open(os.path.join(BASE, "video_list.json"), encoding="utf-8"))
    ev = dedup_evals(json.load(open(os.path.join(BASE, "doctor_evaluations.json"), encoding="utf-8")))
    slots = [
        slot_key(pu, ex)
        for pu in BN_NGHIEN_CUU
        for ex in BAI_NGHIEN_CUU
        if has_doc_eval(ev, pu, ex)
    ]
    if not slots:
        return []
    valid = {slot_key(pu, ex) for pu in BN_NGHIEN_CUU for ex in BAI_NGHIEN_CUU}
    by_slot = {}
    for v in vl:
        sk = slot_key(v.get("username"), v.get("exercise"))
        if sk not in valid:
            continue
        by_slot[sk] = chon_video_moi_hon(v, by_slot[sk]) if sk in by_slot else dict(v)
    out = [by_slot[sk] for sk in slots if sk in by_slot]
    out.sort(key=lambda x: (BN_NGHIEN_CUU.index(x["username"]) if x["username"] in BN_NGHIEN_CUU else 99, x.get("exercise") or ""))
    return out


def is_codman(exercise):
    return "codman" in (exercise or "").lower()


def is_gay(exercise):
    ex = (exercise or "").lower()
    return "gậy" in ex or "gay" in ex or "pulley" in ex


def rom_from_block(block):
    if not isinstance(block, dict):
        block = {}
    return {
        "tb_vai": block.get("tb_goc_vai"),
        "tb_khuyu": block.get("tb_goc_khuyu"),
        "min_vai": block.get("min_goc_vai"),
        "max_vai": block.get("max_goc_vai"),
        "min_khuyu": block.get("min_goc_khuyu"),
        "max_khuyu": block.get("max_goc_khuyu"),
        "std_vai": block.get("std_goc_vai"),
        "std_khuyu": block.get("std_goc_khuyu"),
        "acc": block.get("do_chinh_xac"),
        "pass": block.get("frame_dung"),
        "near": block.get("frame_gan_dung"),
        "fail": block.get("frame_sai"),
        "total": block.get("tong_frame_hop_le"),
    }


def phase_blocks_for_video(v):
    m = v.get("metrics") or {}
    ex = v.get("exercise") or ""
    gay = is_gay(ex)
    cod = is_codman(ex)
    out = []
    if cod:
        for pk, label, tol in PHASE_KEYS:
            pb = phase_block(m, pk, is_gay=False)
            nested = m.get(pk) if isinstance(m.get(pk), dict) else {}
            block = nested if nested.get("min_goc_vai") is not None else m
            if pk == "metrics_g2":
                mc = lay_metrics_chinh(m)
                if mc.get("min_goc_vai") is not None:
                    block = mc
            out.append({
                "phase": label,
                "tolerance": tol,
                **rom_from_block(block),
                "acc": pb.get("acc") if pb.get("acc") is not None else block.get("do_chinh_xac"),
            })
    elif gay and all(isinstance(m.get(pk), dict) and m[pk].get("do_chinh_xac") for pk, _, _ in PHASE_KEYS):
        for pk, label, tol in PHASE_KEYS:
            out.append({"phase": label, "tolerance": tol, **rom_from_block(m[pk])})
    else:
        mc = lay_metrics_chinh(m) or m
        out.append({"phase": "Tổng quan", "tolerance": f"±{v.get('sai_so') or mc.get('sai_so') or 30}°", **rom_from_block(mc)})
    return out


def boxplot_summary_block(v):
    """Số liệu phục vụ mô tả boxplot — lấy GĐ2 Codman hoặc tổng quan Gậy."""
    phases = phase_blocks_for_video(v)
    if is_codman(v.get("exercise")):
        pick = next((p for p in phases if p["phase"] == "G2"), phases[0] if phases else {})
    else:
        pick = phases[0] if phases else {}
    rows = []
    for joint, prefix in (("Vai", "vai"), ("Khuỷu", "khuyu")):
        rows.append({
            "joint": joint,
            "tb": pick.get(f"tb_{prefix}"),
            "min": pick.get(f"min_{prefix}"),
            "max": pick.get(f"max_{prefix}"),
            "std": pick.get(f"std_{prefix}"),
        })
    return rows


def resolve_csv_path(df_path):
    if not df_path:
        return None
    basename = os.path.basename(df_path.replace("\\", "/"))
    candidates = [
        df_path,
        os.path.join(ROOT, df_path.lstrip("/")),
        os.path.join(ROOT, "processed_results", basename),
        os.path.join("/data", "processed_results", basename),
        os.path.join(os.environ.get("DATA_DIR", ""), "processed_results", basename),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def download_csv_hf(basename, token, cache_dir):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Canh bao: chua cai huggingface_hub (pip install huggingface_hub)")
        return None
    if not token:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    last_err = None
    for remote in (f"processed_results/{basename}", basename):
        try:
            path = hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=remote,
                repo_type="dataset",
                token=token,
                local_dir=cache_dir,
            )
            if os.path.isfile(path):
                return path
        except Exception as exc:
            last_err = exc
            continue
    if last_err:
        print(f"Khong tai duoc {basename} tu Dataset: {last_err}")
    return None


def read_display_csv(path):
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        header = pd.read_csv(path, nrows=0)
        cols = [c for c in DISPLAY_CSV_COLS if c in header.columns]
        if not cols:
            return pd.read_csv(path)
        return pd.read_csv(path, usecols=cols)
    except Exception:
        return None


def classify_row(row):
    if row.get("dung"):
        return BOX_GROUPS[0]
    if row.get("gan_dung"):
        return BOX_GROUPS[1]
    return BOX_GROUPS[2]


def boxplot_groups_from_csv(df):
    try:
        import pandas as pd
    except ImportError:
        return None
    if df is None or df.empty:
        return None
    out = {}
    for col, joint in (("goc_vai", "Vai"), ("goc_khuyu", "Khuỷu")):
        if col not in df.columns:
            continue
        plot_df = df.copy()
        if "dung" not in plot_df.columns:
            continue
        plot_df["_grp"] = plot_df.apply(classify_row, axis=1)
        out[joint] = {}
        for grp in BOX_GROUPS:
            sub = plot_df.loc[plot_df["_grp"] == grp, col].dropna()
            if sub.empty:
                out[joint][grp] = {"n": 0}
            else:
                out[joint][grp] = {
                    "n": int(len(sub)),
                    "min": round(float(sub.min()), 1),
                    "q1": round(float(sub.quantile(0.25)), 1),
                    "median": round(float(sub.median()), 1),
                    "q3": round(float(sub.quantile(0.75)), 1),
                    "max": round(float(sub.max()), 1),
                    "mean": round(float(sub.mean()), 1),
                    "std": round(float(sub.std()), 1),
                }
    return out or None


def fmt_num(v, decimals=1):
    if v is None:
        return "—"
    return str(round(float(v), decimals))


def write_txt(videos, payload, csv_mode):
    lines = [
        "ROM + BOXPLOT DATA — 8 video nghiên cứu",
        f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "Nguon: database/video_list.json (metrics giong tab Phan tich tren web)",
        f"Boxplot theo nhom: {'CSV' if csv_mode else 'chi tu JSON (min/max/tb/std GĐ2 hoac tong quan)'}",
        "",
    ]

    lines.append("=" * 72)
    lines.append("A. BIEN DO ROM THEO GIAI DOAN (Codman 3 GD | Gay tong quan hoac 3 GD)")
    lines.append("=" * 72)
    for v in videos:
        lines.append(f"\n{v.get('full_name')} | {v.get('exercise')}")
        lines.append(f"{'GD':<12} | {'Khop':<8} | {'TB':>7} | {'Min':>7} | {'Max':>7} | {'Std':>7} | {'ACC':>7}")
        lines.append("-" * 72)
        for ph in payload[v["username"] + "|" + v["exercise"]]["phases"]:
            for joint, pfx in (("Vai", "vai"), ("Khuỷu", "khuyu")):
                label = f"{ph['phase']} {ph['tolerance']}"
                lines.append(
                    f"{label:<12} | {joint:<8} | {fmt_num(ph.get(f'tb_{pfx}')):>7} | "
                    f"{fmt_num(ph.get(f'min_{pfx}')):>7} | {fmt_num(ph.get(f'max_{pfx}')):>7} | "
                    f"{fmt_num(ph.get(f'std_{pfx}')):>7} | {fmt_num(ph.get('acc')):>6}%"
                )

    lines.append("\n" + "=" * 72)
    lines.append("B. BOXPLOT TOM TAT (GĐ2 Codman | tong quan Gay) — phuc vu bang C.5b")
    lines.append("=" * 72)
    lines.append(f"{'BN':<22} | {'Bai tap':<10} | {'Khop':<8} | {'TB':>7} | {'Min':>7} | {'Max':>7} | {'Std':>7}")
    lines.append("-" * 72)
    for v in videos:
        ex_short = "Codman" if is_codman(v.get("exercise")) else "Gậy"
        for row in payload[v["username"] + "|" + v["exercise"]]["boxplot_summary"]:
            lines.append(
                f"{v.get('full_name'):<22} | {ex_short:<10} | {row['joint']:<8} | "
                f"{fmt_num(row['tb']):>7} | {fmt_num(row['min']):>7} | {fmt_num(row['max']):>7} | {fmt_num(row['std']):>7}"
            )

    any_csv = any(payload[k].get("boxplot_groups") for k in payload)
    if any_csv:
        lines.append("\n" + "=" * 72)
        lines.append("C. BOXPLOT THEO NHOM DUNG / GAN DUNG / SAI (tu CSV)")
        lines.append("=" * 72)
        for v in videos:
            key = v["username"] + "|" + v["exercise"]
            bg = payload[key].get("boxplot_groups")
            if not bg:
                continue
            lines.append(f"\n{v.get('full_name')} | {v.get('exercise')} | CSV: {payload[key].get('csv_path', 'N/A')}")
            for joint, groups in bg.items():
                lines.append(f"  Khop {joint}:")
                for grp, stats in groups.items():
                    if stats.get("n", 0) == 0:
                        lines.append(f"    {grp}: (khong co khung)")
                    else:
                        lines.append(
                            f"    {grp}: n={stats['n']} median={stats['median']} "
                            f"Q1={stats['q1']} Q3={stats['q3']} min={stats['min']} max={stats['max']}"
                        )

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Xuat ROM + Boxplot cho 8 video nghien cuu")
    parser.add_argument("--hf", action="store_true", help="Tai CSV tu HF Dataset neu chua co local")
    parser.add_argument("--csv", action="store_true", help="Co gang doc CSV local/HF de boxplot theo nhom")
    args = parser.parse_args()
    use_csv = args.csv or args.hf
    token = os.environ.get("HF_TOKEN", "").strip() or None
    cache_dir = os.path.join(ROOT, "processed_results")

    videos = load_video_nghien_cuu()
    if len(videos) != 8:
        print(f"Canh bao: tim thay {len(videos)}/8 video nghien cuu")

    payload = {}
    csv_used = False
    for v in videos:
        key = v["username"] + "|" + v["exercise"]
        entry = {
            "full_name": v.get("full_name"),
            "exercise": v.get("exercise"),
            "phases": phase_blocks_for_video(v),
            "boxplot_summary": boxplot_summary_block(v),
            "boxplot_groups": None,
            "csv_path": None,
            "source": "video_list.json",
        }
        if use_csv:
            csv_path = resolve_csv_path(v.get("df_path"))
            if not csv_path and args.hf:
                basename = os.path.basename((v.get("df_path") or "").replace("\\", "/"))
                if basename:
                    csv_path = download_csv_hf(basename, token, cache_dir)
            if csv_path:
                df = read_display_csv(csv_path)
                groups = boxplot_groups_from_csv(df)
                if groups:
                    entry["boxplot_groups"] = groups
                    entry["csv_path"] = csv_path
                    csv_used = True
        payload[key] = entry

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated": datetime.now().isoformat(timespec="seconds"),
                "video_count": len(videos),
                "csv_boxplot": csv_used,
                "videos": payload,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    write_txt(videos, payload, csv_used)
    print(f"Wrote {OUT_TXT}")
    print(f"Wrote {OUT_JSON}")
    if use_csv and not csv_used:
        if args.hf and not token:
            print("Ghi chu: chua co HF_TOKEN.")
            print("  PowerShell: $env:HF_TOKEN = \"hf_xxx\"")
            print("  CMD:        set HF_TOKEN=hf_xxx")
        else:
            print("Ghi chu: khong tim thay CSV local / tai tu HF Dataset that bai.")
            print("  Kiem tra token Read + Dataset quynhphuong1209/Rehab-AI-Monitor-2026-data")


if __name__ == "__main__":
    main()
