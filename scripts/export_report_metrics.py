#!/usr/bin/env python3
"""Xuất số liệu chính xác từ database — dùng cùng logic ưu tiên như app."""
import json
import os
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), "..", "database")
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "_VERIFIED_METRICS.txt")
OUT_PHASES = os.path.join(os.path.dirname(__file__), "..", "docs", "_PHASE_METRICS_TABLES.txt")

PHASE_KEYS = [
    ("metrics_g1", "Giai doan 1 (±45°)"),
    ("metrics_g2", "Giai doan 2 (±30°)"),
    ("metrics_g3", "Giai doan 3 (±15°)"),
]


def parse_t(s):
    if not s:
        return datetime.min
    for fmt in ("%H:%M - %d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except Exception:
            pass
    return datetime.min


def lay_metrics_chinh(m):
    """Metrics chính ±30° — ưu tiên top-level khi nhiều frame hơn metrics_g2."""
    if not isinstance(m, dict):
        return {}
    g2 = m.get("metrics_g2") if isinstance(m.get("metrics_g2"), dict) else {}
    top_tf = int(m.get("tong_frame_hop_le") or 0)
    g2_tf = int(g2.get("tong_frame_hop_le") or 0)
    if m.get("do_chinh_xac") is not None and (not g2_tf or top_tf >= g2_tf):
        return m
    if g2.get("do_chinh_xac") is not None:
        return g2
    return m


def ai_ket_luan(acc):
    if acc >= 80:
        return "Đúng"
    if acc >= 50:
        return "Gần đúng"
    return "Sai"


def rmse_from_mae(mae):
    return round(float(mae) * 1.25, 1) if mae else 0.0


def r1(v):
    return round(float(v), 1) if v is not None else 0.0


def r2(v):
    return round(float(v), 2) if v is not None else 0.0


def phase_block(m, pk, is_gay=False):
    """Lấy metrics một giai đoạn — GĐ2 Codman: g2 nested + override acc/pass nếu top-level mới hơn."""
    nested = m.get(pk) if isinstance(m.get(pk), dict) else {}
    if pk == "metrics_g2" and not is_gay:
        g2 = m.get("metrics_g2") or {}
        mc = lay_metrics_chinh(m)
        top_tf = int(mc.get("tong_frame_hop_le") or 0)
        g2_tf = int(g2.get("tong_frame_hop_le") or 0)
        mg = dict(g2) if g2 else dict(mc)
        # Cao Codman: top-level là lần chạy mới toàn video (ACC khác metrics_g2) — thay hẳn G2
        if (
            g2
            and top_tf > g2_tf * 2
            and mc.get("do_chinh_xac") is not None
            and round(float(mc.get("do_chinh_xac")), 1) != round(float(g2.get("do_chinh_xac") or 0), 1)
        ):
            mg = {**g2, **{k: v for k, v in mc.items() if v is not None}}
        elif top_tf >= g2_tf:
            if mc.get("do_chinh_xac") is not None:
                mg["do_chinh_xac"] = mc["do_chinh_xac"]
            for k in ("mae_tong", "icc", "recall", "precision", "f1_score"):
                if mc.get(k) is not None:
                    mg[k] = mc[k]
    elif is_gay and not nested.get("do_chinh_xac"):
        mg = m
    else:
        mg = nested
    if not mg.get("do_chinh_xac") and pk == "metrics_g2":
        mg = lay_metrics_chinh(m) or m
    return {
        "acc": mg.get("do_chinh_xac"),
        "mae": mg.get("mae_tong"),
        "rmse": rmse_from_mae(mg.get("mae_tong")),
        "icc": mg.get("icc"),
        "recall": mg.get("recall"),
        "precision": mg.get("precision"),
        "f1": mg.get("f1_score"),
        "pass": mg.get("frame_dung"),
        "total": mg.get("tong_frame_hop_le"),
    }


def mean(vals):
    vals = [float(v) for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def mean1(vals):
    return round(mean(vals), 1) if vals else 0.0


def write_metric_table(lines, title, videos, is_gay=False):
    lines.append(title)
    lines.append("-" * len(title))
    for v in videos:
        m = v.get("metrics") or {}
        lines.append(f"\n{v.get('full_name')} | {v.get('exercise')}")
        lines.append(
            f"{'Chi so':<22} | {'G1 ±45°':>10} | {'G2 ±30°':>10} | {'G3 ±15°':>10}"
        )
        blocks = [phase_block(m, pk, is_gay=is_gay) for pk, _ in PHASE_KEYS]
        row_defs = [
            ("ACC (%)", lambda b: f"{r1(b['acc'])}%"),
            ("MAE (°)", lambda b: f"{r1(b['mae'])}"),
            ("RMSE (°)", lambda b: f"{r1(b['rmse'])}"),
            ("ICC", lambda b: f"{r2(b['icc'])}"),
            ("Recall", lambda b: f"{r2(b['recall'])}"),
            ("Precision", lambda b: f"{r2(b['precision'])}"),
            ("F1-Score", lambda b: f"{r2(b['f1'])}"),
            ("Pass (khung)", lambda b: str(int(b["pass"] or 0))),
        ]
        for label, fmt in row_defs:
            vals = [fmt(b) for b in blocks]
            lines.append(f"{label:<22} | {vals[0]:>10} | {vals[1]:>10} | {vals[2]:>10}")

    lines.append(f"\nTRUNG BINH {len(videos)} video:")
    lines.append(
        f"{'Chi so':<22} | {'G1 ±45°':>10} | {'G2 ±30°':>10} | {'G3 ±15°':>10}"
    )
    for label, key, mult, decimals in (
        ("ACC (%)", "acc", 1, 1),
        ("MAE (°)", "mae", 1, 1),
        ("RMSE (°)", "rmse", 1, 1),
        ("ICC", "icc", 1, 2),
        ("Recall", "recall", 1, 2),
        ("Precision", "precision", 1, 2),
        ("F1-Score", "f1", 1, 2),
    ):
        vals = []
        for pk, _ in PHASE_KEYS:
            raw = [phase_block(v.get("metrics") or {}, pk, is_gay=is_gay)[key] for v in videos]
            if key == "rmse":
                raw = [rmse_from_mae(phase_block(v.get("metrics") or {}, pk, is_gay=is_gay)["mae"]) for v in videos]
            m = mean1(raw) if decimals == 1 else mean(raw)
            vals.append(f"{m}%" if key == "acc" else str(m))
        lines.append(f"{label:<22} | {vals[0]:>10} | {vals[1]:>10} | {vals[2]:>10}")
    pass_vals = []
    for pk, _ in PHASE_KEYS:
        pass_vals.append(
            str(sum(int(phase_block(v.get("metrics") or {}, pk, is_gay=is_gay)["pass"] or 0) for v in videos))
        )
    lines.append(f"{'Pass (khung)':<22} | {pass_vals[0]:>10} | {pass_vals[1]:>10} | {pass_vals[2]:>10}")
    lines.append("")


def main():
    vl = json.load(open(os.path.join(BASE, "video_list.json"), encoding="utf-8"))
    ev = json.load(open(os.path.join(BASE, "doctor_evaluations.json"), encoding="utf-8"))

    latest = {}
    for e in ev:
        k = (e.get("patient_username"), e.get("exercise"), e.get("doctor_username"))
        if k not in latest or parse_t(e.get("time")) >= parse_t(latest[k].get("time")):
            latest[k] = e

    codman = [v for v in vl if "Codman" in (v.get("exercise") or "")]
    gay = [v for v in vl if "gậy" in (v.get("exercise") or "").lower()]

    lines = []
    total_frames = 0
    codman_acc = []
    gay_acc = []

    lines.append("VERIFIED FROM database/video_list.json + doctor_evaluations.json")
    lines.append(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    lines.append("RMSE = MAE × 1.25 (cung cong thuc hien thi app)")
    lines.append("GĐ2 Codman: uu tien top-level metrics khi tong_frame >= metrics_g2")
    lines.append("")

    for i, v in enumerate(vl, 1):
        m = v.get("metrics") or {}
        mc = lay_metrics_chinh(m)
        acc = round(float(mc.get("do_chinh_xac") or m.get("do_chinh_xac") or v.get("accuracy") or 0), 1)
        fd = mc.get("frame_dung") or m.get("frame_dung")
        tot = mc.get("tong_frame_hop_le") or m.get("tong_frame_hop_le") or 0
        total_frames += int(tot or 0)

        ex = v.get("exercise") or ""
        if "Codman" in ex:
            codman_acc.append(acc)
        elif "gậy" in ex.lower() or "gay" in ex.lower():
            gay_acc.append(acc)

        g1, g2, g3 = m.get("metrics_g1") or {}, lay_metrics_chinh(m), m.get("metrics_g3") or {}
        kai = (v.get("username"), ex, "AI_Researcher")
        kdoc = (v.get("username"), ex, "doctor1")
        ai_e = latest.get(kai, {})
        doc_e = latest.get(kdoc, {})

        lines.append(f"[{i}] {v.get('full_name')} | {ex}")
        lines.append(f"  video_list.time: {v.get('time')}")
        lines.append(f"  ACC chinh (±30°): {acc}% | frames: {fd}/{tot}")
        lines.append(
            f"  MAE={r1(mc.get('mae_tong') or m.get('mae_tong'))} "
            f"RMSE={rmse_from_mae(mc.get('mae_tong') or m.get('mae_tong'))} "
            f"F1={r2(mc.get('f1_score') or m.get('f1_score'))} "
            f"ICC={r2(mc.get('icc') or m.get('icc'))}"
        )
        lines.append(
            f"  Recall={r2(mc.get('recall'))} Precision={r2(mc.get('precision'))}"
        )
        lines.append(
            f"  goc_vai TB={r1(m.get('tb_goc_vai') or mc.get('tb_goc_vai'))} "
            f"std={r1(m.get('std_goc_vai') or mc.get('std_goc_vai'))}"
        )
        if g1.get("do_chinh_xac"):
            lines.append(
                f"  3GD: G1={r1(g1.get('do_chinh_xac'))}% (Pass {g1.get('frame_dung')}) "
                f"G2={r1(g2.get('do_chinh_xac'))}% (Pass {g2.get('frame_dung')}) "
                f"G3={r1(g3.get('do_chinh_xac'))}% (Pass {g3.get('frame_dung')})"
            )
        else:
            lines.append("  3GD: (chi phan tich tong quan ±30°)")
        lines.append(f"  AI ket luan (rule): {ai_ket_luan(acc)} | AI eval DB: {ai_e.get('doctor_result')} ({ai_e.get('time')})")
        lines.append(f"  BS: {doc_e.get('doctor_result')} | errors={doc_e.get('errors')} ({doc_e.get('time')})")
        lines.append("")

    lines.append("--- TONG KET ---")
    lines.append(f"Tong khung hop le (8 video): {total_frames}")
    if codman_acc:
        lines.append(f"Codman ACC G2: {min(codman_acc)}% - {max(codman_acc)}% TB={round(sum(codman_acc)/len(codman_acc),1)}%")
    if gay_acc:
        lines.append(f"Gay ACC G2: {min(gay_acc)}% - {max(gay_acc)}% TB={round(sum(gay_acc)/len(gay_acc),1)}%")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    phase_lines = [
        f"PHASE METRICS TABLES — Generated {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "RMSE = MAE × 1.25",
        "",
    ]
    write_metric_table(phase_lines, "CODMAN — 4 benh nhan × 3 giai doan", codman)
    write_metric_table(phase_lines, "GAY — 4 benh nhan (Hoa co 3 GD rieng; con lai ±30° tong quan)", gay, is_gay=True)

    with open(OUT_PHASES, "w", encoding="utf-8") as f:
        f.write("\n".join(phase_lines))

    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_PHASES}")


if __name__ == "__main__":
    main()
