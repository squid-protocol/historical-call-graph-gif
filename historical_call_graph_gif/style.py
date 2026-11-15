# Copyright 2026 Joe Esquibel
# SPDX-License-Identifier: Apache-2.0
"""Output style: frame size and framing, titles, quality and timing. Pure data, no rendering."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# name -> settings. Sizes are pixels; pad_* keep text out of areas a platform covers with its own interface.
PRESETS = {
    "default":   dict(size=(1000, 1280), pad_top=0, pad_bottom=0, formats="gif,mp4,html", duration=None),
    "landscape": dict(size=(1920, 1080), pad_top=0, pad_bottom=0, formats="gif,mp4,html", duration=None),
    "square":    dict(size=(1080, 1080), pad_top=0, pad_bottom=0, formats="gif,mp4,html", duration=None),
    "portrait":  dict(size=(1080, 1350), pad_top=0, pad_bottom=0, formats="gif,mp4,html", duration=None),
    # 9:16 for phones: YouTube Shorts, Reels, TikTok. The app draws its own buttons and caption over the
    # bottom fifth and the top bar, so titles and captions stay inside the safe area.
    "shorts":    dict(size=(1080, 1920), pad_top=120, pad_bottom=340, formats="mp4", duration=45.0, text_scale=1.45),
}
ALIASES = {"phone": "shorts", "vertical": "shorts", "reels": "shorts", "tiktok": "shorts", "story": "shorts",
           "youtube": "landscape", "wide": "landscape", "instagram": "portrait"}

# quality -> encoder settings
QUALITY = {
    "draft":    dict(crf=28, x264="veryfast", gif_colors=64,  gif_width=480),
    "standard": dict(crf=23, x264="medium",   gif_colors=128, gif_width=720),
    "high":     dict(crf=19, x264="slow",     gif_colors=256, gif_width=960),
    "max":      dict(crf=15, x264="slower",   gif_colors=256, gif_width=1280),
}

MIN_SIDE, MAX_SIDE = 320, 4096


def preset_names() -> list[str]:
    return sorted(set(PRESETS) | set(ALIASES))


def parse_size(text: str) -> tuple[int, int]:
    """'1080x1920' -> (1080, 1920). Rounded down to even numbers (H.264 requires them)."""
    m = re.fullmatch(r"\s*(\d{3,4})\s*[xX*]\s*(\d{3,4})\s*", text)
    if not m:
        raise ValueError(f"cannot read size {text!r}; use WIDTHxHEIGHT, e.g. 1080x1920")
    w, h = int(m.group(1)) // 2 * 2, int(m.group(2)) // 2 * 2
    if not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
        raise ValueError(f"size must be between {MIN_SIDE} and {MAX_SIDE} pixels on each side")
    return w, h


@dataclass
class Style:
    width: int = 1000
    height: int = 1280
    preset: str = "default"
    pad_top: int = 0
    pad_bottom: int = 0
    title: str | None = None
    subtitle: str | None = None            # replaces the one-line explanation under the legend
    credit: str | None = None              # small text, bottom right (e.g. a channel handle)
    show_legend: bool = True
    show_captions: bool = True
    fit: str = "contain"                   # contain: whole map visible; fill: crop to fill the frame
    zoom: float = 1.0
    focus: str | None = None               # frame a folder or file prefix, e.g. "lib/"
    quality: str = "standard"
    fps: float | None = None               # distinct frames per second of the source (GIF rate)
    text_scale: float = 1.0                # extra text size on top of the frame-size scale (phones read small)
    duration: float | None = None          # seconds of animation (excluding the end hold)
    hold: float = 2.0                      # seconds to linger on the final frame
    frames_per_snapshot: int = 4
    smooth: bool = True                    # MP4: dissolve between rendered frames up to 30 fps
    gif_width: int | None = None
    formats: str = "gif,mp4,html"
    out_fps: int = 30
    notes: list = field(default_factory=list)

    @property
    def scale(self) -> float:
        """Design scale: text and line widths were designed for a 1000 px short side."""
        return min(self.width, self.height) / 1000.0

    @property
    def q(self) -> dict:
        return QUALITY[self.quality]

    @property
    def suffix(self) -> str:
        return "" if self.preset in ("default", "custom") else f"-{self.preset}"


def resolve(preset: str | None = None, size: str | None = None, **kw) -> Style:
    """Combine a preset, an explicit size and individual options into one validated Style."""
    name = ALIASES.get(preset or "default", preset or "default")
    if name not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {', '.join(preset_names())}")
    p = PRESETS[name]
    st = Style(width=p["size"][0], height=p["size"][1], preset=name, pad_top=p["pad_top"], pad_bottom=p["pad_bottom"],
               formats=p["formats"], duration=p["duration"], text_scale=p.get("text_scale", 1.0))
    if size:
        st.width, st.height = parse_size(size)
        if name != "default":
            st.notes.append(f"size {st.width}x{st.height} overrides the {name} preset's size")
        st.preset = "custom" if name == "default" else name
    for k, v in kw.items():
        if v is None:
            continue
        if not hasattr(st, k):
            raise ValueError(f"unknown option {k}")
        setattr(st, k, v)
    if st.quality not in QUALITY:
        raise ValueError(f"quality must be one of {', '.join(QUALITY)}")
    if st.fit not in ("contain", "fill"):
        raise ValueError("fit must be 'contain' or 'fill'")
    if not (0.25 <= st.zoom <= 8):
        raise ValueError("zoom must be between 0.25 and 8")
    if st.duration is not None and not (3 <= st.duration <= 600):
        raise ValueError("duration must be between 3 and 600 seconds")
    if not (0.5 <= st.text_scale <= 3):
        raise ValueError("text scale must be between 0.5 and 3")
    if st.hold < 0 or st.hold > 30:
        raise ValueError("hold must be between 0 and 30 seconds")
    if not (1 <= st.frames_per_snapshot <= 24):
        raise ValueError("frames per snapshot must be between 1 and 24")
    if st.gif_width is None:
        st.gif_width = min(st.q["gif_width"], st.width)
    st.gif_width = st.gif_width // 2 * 2
    if st.preset == "shorts" and st.duration and st.duration + st.hold > 180:
        st.notes.append("YouTube Shorts can be at most 3 minutes long")
    return st
