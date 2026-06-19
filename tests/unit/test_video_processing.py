from types import SimpleNamespace

import numpy as np

from video.processing import xu_ly_frame


class _FakeCv2:
    COLOR_BGR2RGB = 1
    FONT_HERSHEY_DUPLEX = 0

    def cvtColor(self, frame, _code):
        return frame

    def rectangle(self, *_args, **_kwargs):
        return None

    def addWeighted(self, overlay, *_args, **_kwargs):
        return overlay

    def putText(self, *_args, **_kwargs):
        return None


def test_xu_ly_frame_without_pose_returns_empty_analysis():
    deps = SimpleNamespace(cv2=_FakeCv2(), gc=__import__("gc"))
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    model = SimpleNamespace(process=lambda _rgb: SimpleNamespace(pose_landmarks=None))

    output, goc_vai, goc_khuyu, dung, eval_info, warnings, landmarks = xu_ly_frame(
        deps,
        frame,
        model,
        {"vai": 90, "khuyu": 170, "sai_so": 10},
        0,
    )

    assert output.shape == frame.shape
    assert goc_vai is None
    assert goc_khuyu is None
    assert dung is None
    assert eval_info is None
    assert warnings == []
    assert landmarks is None
