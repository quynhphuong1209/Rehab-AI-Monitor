import pandas as pd

from video.metrics import recalc_metrics, segment_frames, tinh_metrics_chi_tiet


def test_segment_frames_short_input_splits_evenly():
    assert segment_frames([{"goc_vai": 90, "goc_khuyu": 170}] * 12) == [0, 4, 8, 12]


def test_recalc_metrics_counts_exact_and_near_frames():
    df = pd.DataFrame(
        [
            {"goc_vai": 90, "goc_khuyu": 170, "vai_chuan": 90, "khuyu_chuan": 170},
            {"goc_vai": 102, "goc_khuyu": 182, "vai_chuan": 90, "khuyu_chuan": 170},
            {"goc_vai": 130, "goc_khuyu": 130, "vai_chuan": 90, "khuyu_chuan": 170},
        ]
    )

    metrics = recalc_metrics(df, 10, "codman")

    assert metrics["frame_dung"] == 1
    assert metrics["frame_gan_dung"] == 1
    assert metrics["frame_sai"] == 1
    assert metrics["tong_frame_hop_le"] == 3


def test_tinh_metrics_chi_tiet_uses_boolean_columns():
    df = pd.DataFrame(
        [
            {
                "goc_vai": 90,
                "goc_khuyu": 170,
                "vai_chuan": 90,
                "khuyu_chuan": 170,
                "dung": True,
                "gan_dung": True,
                "vai_dung": True,
                "khuyu_dung": True,
            },
            {
                "goc_vai": 100,
                "goc_khuyu": 180,
                "vai_chuan": 90,
                "khuyu_chuan": 170,
                "dung": False,
                "gan_dung": True,
                "vai_dung": False,
                "khuyu_dung": True,
            },
        ]
    )

    metrics = tinh_metrics_chi_tiet(df, {"ten": "codman"})

    assert metrics["frame_dung"] == 1
    assert metrics["frame_gan_dung"] == 1
    assert metrics["ty_le_tong_the"] == 50.0
    assert metrics["ty_le_gan_dung"] == 50.0
