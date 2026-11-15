# Copyright 2026 Joe Esquibel
# SPDX-License-Identifier: Apache-2.0
"""GIF / MP4 / PNG renderer: matplotlib frames, assembled with ffmpeg when present.

Every position is computed from the frame size in ``Style``, so the same code draws a 1920x1080 landscape
video, a 1080x1920 phone video (YouTube Shorts, Reels, TikTok) or a square image.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import shutil
import subprocess
import tempfile
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np

from .layout import dir_of
from .scene import Scene, age_full_years, road_age_years, road_ages, snap_years
from .style import Style

BG = "#0b0f17"
BIRTH, DEATH = "#d2f56a", "#ffb000"
AGE_COLORS = ["#d2f56a", "#38d6a4", "#3b86f0"]      # new -> old: lime, teal, blue (colour-blind safe ramp)
PT_PER_PX = 72 / 100                                  # figures are drawn at 100 dpi

_S: Scene | None = None          # set in the parent before forking workers
_ST: Style | None = None
_CACHE: dict = {}


def _rgba(h, a):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)


def _lw(n: int) -> float:
    return min(2.2, 0.35 + 0.25 * np.sqrt(n))


def _prepare(sc: Scene) -> dict:
    ds, lay = sc.ds, sc.lay
    n = sc.n
    alive = []
    for s in range(n):
        v = np.zeros(len(ds.funcs), bool)
        for i in ds.alive[s]:
            if lay.func_file[i] >= 0:
                v[i] = True
        alive.append(v)
    indeg = []
    for s in range(n):
        c = Counter(b for _, b in ds.calls[s])
        arr = np.zeros(len(ds.funcs))
        for k, v in c.items():
            arr[k] = v
        indeg.append(arr)
    ages = [road_ages(r.counts) for r in sc.roads]
    years = snap_years(ds)
    ageyrs = [road_age_years(r.counts, years) for r in sc.roads]
    full = age_full_years(years)
    wt = Counter()
    for r in sc.roads:
        w = max(r.counts)
        wt[r.a] += w
        wt[r.b] += w
    hubs = [p for p, _ in wt.most_common(6)]
    groups = {}
    for i, p in enumerate(lay.files):
        groups.setdefault(dir_of(p), []).append(i)
    dcen = {d: lay.file_xy[fs].mean(0) for d, fs in groups.items()}
    return dict(alive=alive, indeg=indeg, ages=ages, ageyrs=ageyrs, full=full, hubs=hubs, dcen=dcen, groups=groups)


def _text_scale(st: Style) -> float:
    return st.scale * st.text_scale


def _fit_pt(text: str, pt: float, avail_px: float, floor: float) -> float:
    """Largest font size <= pt whose estimated width fits ``avail_px``."""
    est = len(text) * 0.60 / PT_PER_PX
    return max(floor, min(pt, avail_px / max(est, 1)))


def _note_lines(st: Style) -> list[str]:
    note = st.subtitle or "Circles are files, dots are functions (brighter = more callers). Only what is alive that snapshot is shown."
    T = _text_scale(st)
    per_line = max(20, int((st.width - 40 * T) / (10.5 * T * 0.60 / PT_PER_PX)))
    return textwrap.wrap(note, per_line) or [""]


def _header_height(st: Style) -> float:
    T = _text_scale(st)
    h = st.pad_top + 12 * T + 36 * T + 36 * T            # title, stats row
    if st.show_legend:
        h += 62 * T + 18 * T * len(_note_lines(st))      # legend block, explanation lines
    return h + 14 * T


def geometry(sc: Scene, st: Style) -> dict:
    """Map placement: pixels per data unit and the data coordinates visible in the frame."""
    lay = sc.lay
    W, H, T = st.width, st.height, _text_scale(st)
    top = _header_height(st)
    bottom = st.pad_bottom + (150 * T if st.show_captions else 40 * T)
    free_h, free_w = max(H - top - bottom, 100), W
    fxy, fr = lay.file_xy, lay.file_r
    if len(fr):                       # fit to the layout's real outline (trimming a few far outliers), not a circle
        x0, x1 = np.percentile(fxy[:, 0] - fr, 0.5), np.percentile(fxy[:, 0] + fr, 99.5)
        y0, y1 = np.percentile(fxy[:, 1] - fr, 0.5), np.percentile(fxy[:, 1] + fr, 99.5)
        cx, cy, ex, ey = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, 5), max((y1 - y0) / 2, 5)
    else:
        cx = cy = 0.0
        ex = ey = lay.extent
    if st.focus:
        pre = st.focus.lstrip("./")
        idx = [i for i, p in enumerate(lay.files) if p.startswith(pre)]
        if not idx:
            raise RuntimeError(f"--focus {st.focus!r} matches no files")
        x0 = min(lay.file_xy[i][0] - lay.file_r[i] for i in idx)
        x1 = max(lay.file_xy[i][0] + lay.file_r[i] for i in idx)
        y0 = min(lay.file_xy[i][1] - lay.file_r[i] for i in idx)
        y1 = max(lay.file_xy[i][1] + lay.file_r[i] for i in idx)
        cx, cy, ex, ey = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, 5), max((y1 - y0) / 2, 5)
    kx, ky = free_w / (2 * ex * 1.04), free_h / (2 * ey * 1.04)
    k = (min(kx, ky) if st.fit == "contain" else max(kx, ky)) * st.zoom
    y_center_px = top + free_h / 2
    y_top = cy + y_center_px / k
    return dict(k=k, xlim=(cx - W / (2 * k), cx + W / (2 * k)), ylim=(y_top - H / k, y_top), top=top, bottom=bottom)


def _figure(st: Style):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "fig" not in _CACHE:
        _CACHE["fig"] = plt.figure(figsize=(st.width / 100, st.height / 100), dpi=100, facecolor=BG)
    return _CACHE["fig"]


def render_frame(args) -> str:
    s, t, path = args
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    sc, st = _S, _ST
    if "prep" not in _CACHE:
        _CACHE["prep"] = _prepare(sc)
        _CACHE["geo"] = geometry(sc, st)
    P, geo = _CACHE["prep"], _CACHE["geo"]
    cm = LinearSegmentedColormap.from_list("age", AGE_COLORS)
    lay, ds = sc.lay, sc.ds
    W, H, S, T = st.width, st.height, st.scale, _text_scale(st)
    k = geo["k"]
    title = st.title or f"Roads of {ds.title}"
    fig = _figure(st)
    fig.clf()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(*geo["xlim"])
    ax.set_ylim(*geo["ylim"])
    fade = max(0.0, 1 - t)
    alive = P["alive"][s]
    prev = P["alive"][s - 1] if s else alive
    born, died = np.where(alive & ~prev)[0], np.where(prev & ~alive)[0]
    stay = np.where(alive & ~(alive & ~prev))[0]
    pxs = lambda r: (r * 2 * k) ** 2 * 0.5                 # data radius -> marker area
    fx, fr = lay.file_xy, lay.file_r
    live_files = set(int(lay.func_file[i]) for i in np.where(alive | prev)[0])
    for f in live_files:
        ax.add_patch(plt.Circle(fx[f], fr[f], fill=False, ec=_rgba("#8a96b0", 0.22), lw=0.35 * S, zorder=0.3))
    # roads
    items = [(r, P["ages"][j][s], P["ageyrs"][j][s]) for j, r in enumerate(sc.roads) if r.counts[s]]
    items.sort(key=lambda x: x[2])
    segs, cols, lws = [], [], []
    for r, age, yrs in items:
        a = min(yrs / P["full"], 1.0)
        c = to_rgb(cm(a))
        al = (0.20 + 0.30 * min(1, np.sqrt(r.counts[s]) / 4)) * (1 - 0.35 * a)
        if age <= 1 and s > 0:
            al = min(1, al + 0.30 * fade)
        segs.append(r.poly)
        cols.append((*c, al))
        lws.append(_lw(r.counts[s]) * S)
    if segs:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=1.0, capstyle="round", joinstyle="round"))
    gone = [r for r in sc.roads if s and r.counts[s - 1] and not r.counts[s]]
    if gone:
        ax.add_collection(LineCollection([r.poly for r in gone], colors=[_rgba(DEATH, 0.45 * fade)],
                                         linewidths=[_lw(r.counts[s - 1]) * S for r in gone], zorder=1.1, capstyle="round"))
    # functions
    xy, rr = lay.func_xy, lay.func_r
    if len(stay):
        g = [min(0.95, 0.30 + 0.2 * np.log1p(P["indeg"][s][i])) for i in stay]
        ax.scatter(xy[stay, 0], xy[stay, 1], s=pxs(rr[stay]) * 2.0, c=[_rgba("#c4cde0", a) for a in g], linewidths=0, zorder=2.1)
    if len(born):
        ax.scatter(xy[born, 0], xy[born, 1], s=pxs(rr[born]) * (2.0 + 2.6 * fade), c=[_rgba(BIRTH, 0.40 + 0.55 * fade)], linewidths=0, zorder=2.2)
    if len(died):
        ax.scatter(xy[died, 0], xy[died, 1], s=pxs(rr[died]) * (2.0 + 1.6 * fade), c=[_rgba(DEATH, 0.85 * fade)], linewidths=0, zorder=2.2)
    # data-space labels with collision avoidance
    halo = [pe.withStroke(linewidth=2.5 * S, foreground=BG)]
    boxes = []

    def place(txt, x, y, fs, **kw):
        w, h = len(txt) * fs * 0.58 / PT_PER_PX / k, fs * 1.5 / PT_PER_PX / k
        b = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        if not (geo["xlim"][0] < x < geo["xlim"][1] and geo["ylim"][0] < y < geo["ylim"][1]):
            return
        if any(b[0] < q[2] and b[2] > q[0] and b[1] < q[3] and b[3] > q[1] for q in boxes):
            return
        boxes.append(b)
        ax.text(x, y, txt, fontsize=fs, ha="center", va="center", zorder=5, path_effects=halo, **kw)

    lab = 9 * S * st.text_scale
    for f in P["hubs"]:
        if f in live_files:
            place(lay.files[f].split("/")[-1], fx[f][0], fx[f][1] - fr[f] - 14 / k, lab * 0.95, color="#f6ecd2")
    ranked = sorted(P["dcen"].items(), key=lambda kv: -sum(1 for f in P["groups"][kv[0]] if f in live_files))
    for d, c in ranked:
        if sum(1 for f in P["groups"][d] if f in live_files) >= 4:
            place(d, c[0], c[1], lab, color="#9aa6bf", alpha=0.8)
    for s0 in range(max(0, s - 2), s + 1):
        age = (s - s0) + t
        if age < 3:
            fa = 1 - age / 3
            for d, kk, files in sc.new_dirs[s0]:
                for f in files:
                    ax.add_patch(plt.Circle(fx[f], fr[f] + 1.2, fill=False, ec=_rgba("#ffffff", 0.3 + 0.6 * fa), lw=(1.0 + 1.6 * fa) * S, zorder=6))
    # ---- overlay in pixel coordinates (y grows downwards): header, legend, captions
    ov = fig.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, W)
    ov.set_ylim(H, 0)
    ov.axis("off")
    ov.patch.set_alpha(0)
    outl = [pe.withStroke(linewidth=4 * T, foreground="#000000")]
    sx = 20 * T
    y = st.pad_top + 12 * T
    st_row = sc.stats[s]
    avail = W - 2 * sx
    parts = [ds.snaps[s].label, f"{st_row['funcs']:,} functions", f"{st_row['calls']:,} calls", f"{st_row['roads']:,} roads (file to file)"]
    for n_parts in (4, 3, 2):            # drop the least important figures before shrinking the text on a narrow frame
        stats = "   |   ".join(parts[:n_parts])
        if len(stats) * 0.60 / PT_PER_PX * (16 * T * 0.85) <= avail or n_parts == 2:     # width at 85% of full size
            break
    ov.text(sx, y, title, color="#f2f5fb", fontsize=_fit_pt(title, 25 * T, avail, 10), va="top", weight="bold")
    y += 36 * T
    ov.text(sx, y, stats, color="#c4cde0", fontsize=_fit_pt(stats, 16 * T, avail, 8), va="top")
    y += 36 * T
    if st.show_legend:
        gx0, gx1 = sx, sx + 160 * T
        ov.imshow(np.linspace(0, 1, 128)[None], extent=(gx0, gx1, y + 24 * T, y + 10 * T), aspect="auto", cmap=cm, zorder=3)
        ov.set_xlim(0, W)
        ov.set_ylim(H, 0)
        old = f"{P['full']:.0f}+ yr" if P["full"] >= 1.5 else "old"
        ov.text(gx0, y + 28 * T, "new", color=AGE_COLORS[0], fontsize=11 * T, va="top")
        ov.text(gx1, y + 28 * T, old, color=AGE_COLORS[2], fontsize=11 * T, va="top", ha="right")
        lx = gx1 + 24 * T
        ov.text(lx, y + 2 * T, "road colour = age of the road", color="#d5dcec", fontsize=12 * T, va="top")
        ov.text(lx, y + 24 * T, "road width = calls merged into it", color="#d5dcec", fontsize=12 * T, va="top")
        bx = lx + 360 * T if W >= 1400 or T < 1.3 else W - 150 * T
        ov.text(bx, y, "●", color=DEATH, fontsize=15 * T, va="top")
        ov.text(bx + 26 * T, y + 2 * T, "died", color="#d5dcec", fontsize=12 * T, va="top")
        ov.text(bx, y + 22 * T, "●", color=BIRTH, fontsize=15 * T, va="top")
        ov.text(bx + 26 * T, y + 24 * T, "born", color="#d5dcec", fontsize=12 * T, va="top")
        y += 62 * T
        for line in _note_lines(st):
            ov.text(sx, y, line, color="#9aa6bf", fontsize=10.5 * T, va="top")
            y += 18 * T
    # bottom: footer, caption, new-folder callout, credit
    base = H - st.pad_bottom
    foot = "A file that moves shows as a death plus a birth."
    if lay.dropped_files:
        foot += f"   Not drawn: {lay.dropped_files:,} small files ({lay.dropped_funcs:,} functions)."
    ov.text(sx, base - 8 * T, foot, color="#7f8aa3", fontsize=_fit_pt(foot, 9.5 * T, avail, 6), va="bottom")
    if st.credit:
        ov.text(W - sx, base - 8 * T, st.credit, color="#9aa6bf", fontsize=11 * T, va="bottom", ha="right")
    if st.show_captions:
        per = int(max(30, min(96, avail / (14 * T * 0.58 / PT_PER_PX))))
        lines = textwrap.wrap(sc.captions[s], per)
        line_h = 14 * T * 1.25 / PT_PER_PX
        top_y = base - 30 * T - len(lines) * line_h
        ov.text(W / 2, base - 30 * T, "\n".join(lines), color="#ffffff", fontsize=14 * T, ha="center", va="bottom",
                linespacing=1.25, weight="bold", path_effects=outl)
        fresh = []
        for s0 in range(max(0, s - 2), s + 1):
            if (s - s0) + t < 3:
                fresh += [(d, kk) for d, kk, _ in sc.new_dirs[s0][:2]]
        if fresh:
            msg = "NEW FOLDER: " + "   |   ".join(d for d, _ in fresh[:2])
            ov.text(W / 2, top_y - 8 * T, msg, color="#ffe9a8", fontsize=_fit_pt(msg, 14 * T, avail, 8), ha="center", va="bottom",
                    weight="bold", path_effects=outl)
    fig.savefig(path, facecolor=BG)
    return path


def _ffmpeg_mp4(ff, pat, dst, st: Style, fps_in: float) -> None:
    chain = []
    if st.hold > 0:                       # pad first: after the blend filter the padding is silently dropped
        chain.append(f"tpad=stop_mode=clone:stop_duration={st.hold}")
    if st.smooth and fps_in < st.out_fps:
        chain.append(f"framerate=fps={st.out_fps}:interp_start=64:interp_end=192:scene=100")   # quick dissolve between frames (a long one would double-expose the captions)
    else:
        chain.append(f"fps={st.out_fps}")
    chain.append("format=yuv420p")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", f"{fps_in:.4f}", "-i", pat, "-vf", ",".join(chain),
                    "-c:v", "libx264", "-preset", st.q["x264"], "-crf", str(st.q["crf"]), "-movflags", "+faststart",
                    "-r", str(st.out_fps), str(dst)], check=True)


def _ffmpeg_gif(ff, pat, dst, st: Style, fps_in: float, tmp: Path) -> None:
    pad = f"tpad=stop_mode=clone:stop_duration={st.hold}," if st.hold > 0 else ""
    sc = f"scale={st.gif_width}:-2:flags=lanczos"
    pal = tmp / "pal.png"
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", f"{fps_in:.4f}", "-i", pat, "-vf",
                    f"{pad}{sc},palettegen=max_colors={st.q['gif_colors']}:stats_mode=diff", str(pal)], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", f"{fps_in:.4f}", "-i", pat, "-i", str(pal), "-lavfi",
                    f"{pad}{sc}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle", str(dst)], check=True)


def render(sc: Scene, out_dir: Path, name: str = "roads", style: Style | None = None, jobs: int = 3,
           progress=print) -> list[Path]:
    global _S, _ST
    st = style or Style()
    _S, _ST = sc, st
    _CACHE.clear()
    geometry(sc, st)                                           # fail early on a bad --focus
    formats = {f.strip() for f in st.formats.split(",") if f.strip()}
    visual = formats & {"gif", "mp4", "png", "frames"}
    made: list[Path] = []
    if not visual:
        return made
    tmp = Path(tempfile.mkdtemp(prefix="hcg_frames_"))
    sub = st.frames_per_snapshot
    jobs_list = [(s, u / sub, str(tmp / f"f{s * sub + u:05d}.png")) for s in range(sc.n) for u in range(sub)]
    n_frames = len(jobs_list)
    fps_in = st.fps or (n_frames / st.duration if st.duration else 6.0)
    progress(f"  rendering {n_frames} frames at {st.width}x{st.height}"
             + (f" ({st.preset} preset)" if st.preset != "default" else ""))
    for note in st.notes:
        progress(f"  note: {note}")
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = None
    try:
        if ctx and jobs > 1 and n_frames > 8:
            with ctx.Pool(jobs) as pool:
                for _ in pool.imap(render_frame, jobs_list, chunksize=2):
                    pass
        else:
            for j in jobs_list:
                render_frame(j)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{name}{st.suffix}"
        ff = shutil.which("ffmpeg")
        pat = str(tmp / "f%05d.png")
        if "mp4" in formats:
            if ff:
                dst = out_dir / f"{stem}.mp4"
                _ffmpeg_mp4(ff, pat, dst, st, fps_in)
                made.append(dst)
            else:
                progress("  ffmpeg not found: skipping MP4 (install ffmpeg to enable it)")
        if "gif" in formats:
            dst = out_dir / f"{stem}.gif"
            gif_fps = min(fps_in, 50.0)
            if ff:
                _ffmpeg_gif(ff, pat, dst, st, gif_fps, tmp)
            else:
                from PIL import Image
                frames = []
                for j in jobs_list:
                    im = Image.open(j[2]).convert("RGB")
                    im = im.resize((st.gif_width, int(im.height * st.gif_width / im.width)), Image.LANCZOS)
                    frames.append(im.quantize(colors=st.q["gif_colors"], method=Image.MEDIANCUT))
                dur = [int(1000 / gif_fps)] * len(frames)
                dur[-1] += int(st.hold * 1000)
                frames[0].save(dst, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=False)
            made.append(dst)
        if "png" in formats:
            dst = out_dir / f"{stem}.png"
            render_frame((sc.n - 1, 1.0, str(dst)))             # the settled final state, no flash
            made.append(dst)
        if "frames" in formats:
            d = out_dir / f"{stem}-frames"
            d.mkdir(exist_ok=True)
            for j in jobs_list:
                shutil.copy(j[2], d / Path(j[2]).name)
            made.append(d)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return made
