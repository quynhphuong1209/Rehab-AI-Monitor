#!/usr/bin/env python3
"""Đối chiếu số liệu báo cáo với database/*.json (video_list, doctor_evaluations)."""
import json
import os
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), "..", "database")


def pseudonym(value, fallback="Participant"):
    raw = str(value or "").strip()
    if not raw:
        return fallback
    import hashlib
    return f"{fallback}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]}"


def parse_t(s):
    if not s:
        return datetime.min
    try:
        return datetime.strptime(s.strip(), "%H:%M - %d/%m/%Y")
    except Exception:
        return datetime.min


def main():
    vl = json.load(open(os.path.join(BASE, "video_list.json"), encoding="utf-8"))
    ev = json.load(open(os.path.join(BASE, "doctor_evaluations.json"), encoding="utf-8"))

    latest = {}
    for e in ev:
        k = (e.get("patient_username"), e.get("exercise"), e.get("doctor_username"))
        if k not in latest or parse_t(e.get("time")) >= parse_t(latest[k].get("time")):
            latest[k] = e

    total_frames = 0
    codman_g2 = []
    gay_g2 = []

    print("=" * 80)
    print("NGUON: database/video_list.json + doctor_evaluations.json (latest eval)")
    print("=" * 80)

    for i, v in enumerate(vl, 1):
        m = v.get("metrics") or {}
        g1 = m.get("metrics_g1") or {}
        g2 = m.get("metrics_g2") or {}
        g3 = m.get("metrics_g3") or {}
        tf = g2.get("tong_frame_hop_le") or 0
        total_frames += tf

        name = pseudonym(v.get("username"), "BN")
        ex = v.get("exercise")
        acc_g2 = g2.get("do_chinh_xac")
        if "Codman" in ex:
            codman_g2.append(acc_g2)
        elif "gậy" in ex.lower() or "gay" in ex.lower():
            gay_g2.append(acc_g2)

        kai = (v.get("username"), ex, "AI_Researcher")
        kdoc = (v.get("username"), ex, "doctor1")
        ai_e = latest.get(kai, {})
        doc_e = latest.get(kdoc, {})

        print(f"\n[{i}] {name} | {ex}")
        print(f"  video_list.time: {v.get('time')}")
        print(f"  G1: {g1.get('do_chinh_xac')}% ({g1.get('frame_dung')}/{g1.get('tong_frame_hop_le')})")
        print(f"  G2: {g2.get('do_chinh_xac')}% ({g2.get('frame_dung')}/{g2.get('tong_frame_hop_le')})")
        print(f"  G3: {g3.get('do_chinh_xac')}% ({g3.get('frame_dung')}/{g3.get('tong_frame_hop_le')})")
        print(f"  MAE={round(g2.get('mae_tong') or 0, 1)} F1={round(g2.get('f1_score') or 0, 2)} ICC={round(g2.get('icc') or 0, 2)}")
        print(f"  tb_goc_vai={round(m.get('tb_goc_vai') or 0, 1)} tb_goc_khuyu={round(m.get('tb_goc_khuyu') or 0, 1)}")
        print(f"  AI latest present: {'yes' if ai_e else 'no'}")
        print(f"  Clinical latest present: {'yes' if doc_e else 'no'}")

    print(f"\n--- TONG KET ---")
    print(f"Tong frames (sum G2.tong_frame_hop_le): {total_frames}")
    if codman_g2:
        print(f"Codman G2 ACC range: {min(codman_g2):.1f}% - {max(codman_g2):.1f}% TB={sum(codman_g2)/len(codman_g2):.1f}%")
    if gay_g2:
        print(f"Gay G2 ACC range: {min(gay_g2):.1f}% - {max(gay_g2):.1f}% TB={sum(gay_g2)/len(gay_g2):.1f}%")


if __name__ == "__main__":
    main()
