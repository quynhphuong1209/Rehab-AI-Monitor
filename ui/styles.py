"""CSS loading helpers for Streamlit UI assets."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=16)
def read_css_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        print(f"[CSS] Khong doc duoc {path}: {exc}")
        return ""


def load_css_file(st, path: str) -> None:
    css = read_css_file(path)
    if css:
        st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)


def inject_base_css(st, *, assets_dir: str = "assets") -> None:
    """Inject CSS assets each rerun because Streamlit rebuilds the DOM."""
    load_css_file(st, os.path.join(assets_dir, "styles.css"))
    theme_file = "theme_light.css" if st.session_state.get("theme") == "light" else "theme_dark.css"
    load_css_file(st, os.path.join(assets_dir, theme_file))
