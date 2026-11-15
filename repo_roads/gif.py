"""GIF / MP4 renderer (matplotlib frames, assembled with ffmpeg when present)."""
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
from .scene import AGE_FULL, Scene, road_ages

BG = "#0b0f17"
BIRTH, DEATH = "#d2f56a", "#ffb000"
AGE_COLORS = ["#d2f56a", "#38d6a4", "#3b86f0"]      # new -> old: lime, teal, blue (colour-blind safe ramp)

_S: Scene | None = None          # set in the parent before forking workers
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
    wt = Counter()
    for r in sc.roads:
        w = max(r.counts)
        wt[r.a] += w
        wt[r.b] += w
    hubs = [p for p, _ in wt.most_common(6)]
    dcen = {}
    groups = {}
    for i, p in enumerate(lay.files):
        groups.setdefault(dir_of(p), []).append(i)
    for d, fs in groups.items():
        dcen[d] = lay.file_xy[fs].mean(0)
    return dict(alive=alive, indeg=indeg, ages=ages, hubs=hubs, dcen=dcen, groups=groups)


def _figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "fig" not in _CACHE:
        _CACHE["fig"] = plt.figure(figsize=(10, 12.8), dpi=100, facecolor=BG)
    return _CACHE["fig"]


def render_frame(args) -> str:
    s, t, path, title = args
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    sc = _S
    if "prep" not in _CACHE:
        _CACHE["prep"] = _prepare(sc)
    P = _CACHE["prep"]
    cm = LinearSegmentedColormap.from_list("age", AGE_COLORS)
    lay, ds = sc.lay, sc.ds
    ext = lay.extent
    fig = _figure()
    fig.clf()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-ext, ext)
    ax.set_ylim(-ext * 1.22, ext * 1.34)
    fade = max(0.0, 1 - t)
    alive = P["alive"][s]
    prev = P["alive"][s - 1] if s else alive
    born, died = np.where(alive & ~prev)[0], np.where(prev & ~alive)[0]
    stay = np.where(alive & ~(alive & ~prev))[0]
    pxs = lambda r: (r * 2 * (fig.get_size_inches()[0] * fig.dpi) / (2 * ext)) ** 2 * 0.5
    fx, fr = lay.file_xy, lay.file_r
    live_files = set(int(lay.func_file[i]) for i in np.where(alive | prev)[0])
    for f in live_files:
        ax.add_patch(plt.Circle(fx[f], fr[f], fill=False, ec=_rgba("#8a96b0", 0.22), lw=0.35, zorder=0.3))
    # roads
    items = [(r, P["ages"][k][s]) for k, r in enumerate(sc.roads) if r.counts[s]]
    items.sort(key=lambda x: x[1])
    segs, cols, lws = [], [], []
    for r, age in items:
        a = min(age / AGE_FULL, 1.0)
        c = to_rgb(cm(a))
        al = (0.20 + 0.30 * min(1, np.sqrt(r.counts[s]) / 4)) * (1 - 0.35 * a)
        if age <= 1 and s > 0:
            al = min(1, al + 0.30 * fade)
        segs.append(r.poly)
        cols.append((*c, al))
        lws.append(_lw(r.counts[s]))
    if segs:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=1.0, capstyle="round", joinstyle="round"))
    gone = [r for r in sc.roads if s and r.counts[s - 1] and not r.counts[s]]
    if gone:
        ax.add_collection(LineCollection([r.poly for r in gone], colors=[_rgba(DEATH, 0.45 * fade)],
                                         linewidths=[_lw(r.counts[s - 1]) for r in gone], zorder=1.1, capstyle="round"))
    # functions
    xy, rr = lay.func_xy, lay.func_r
    if len(stay):
        g = [min(0.95, 0.30 + 0.2 * np.log1p(P["indeg"][s][i])) for i in stay]
        ax.scatter(xy[stay, 0], xy[stay, 1], s=pxs(rr[stay]) * 2.0, c=[_rgba("#c4cde0", a) for a in g], linewidths=0, zorder=2.1)
    if len(born):
        ax.scatter(xy[born, 0], xy[born, 1], s=pxs(rr[born]) * (2.0 + 2.6 * fade), c=[_rgba(BIRTH, 0.40 + 0.55 * fade)], linewidths=0, zorder=2.2)
    if len(died):
        ax.scatter(xy[died, 0], xy[died, 1], s=pxs(rr[died]) * (2.0 + 1.6 * fade), c=[_rgba(DEATH, 0.85 * fade)], linewidths=0, zorder=2.2)
    T = ax.transAxes
    out = [pe.withStroke(linewidth=4, foreground="#000000")]
    halo = [pe.withStroke(linewidth=2.5, foreground=BG)]
    st = sc.stats[s]
    ax.text(0.02, 0.99, title, color="#f2f5fb", fontsize=25, va="top", weight="bold", transform=T)
    ax.text(0.02, 0.962, f"{ds.snaps[s].label}   |   {st['funcs']:,} functions   |   {st['calls']:,} calls   |   {st['roads']:,} roads (file to file)",
            color="#c4cde0", fontsize=16, va="top", transform=T)
    gx = fig.add_axes([0.02, 0.915, 0.16, 0.012])
    gx.imshow(np.linspace(0, 1, 128)[None], aspect="auto", cmap=cm)
    gx.axis("off")
    ax.text(0.02, 0.908, "new", color=AGE_COLORS[0], fontsize=11, va="top", transform=T)
    ax.text(0.18, 0.908, "old", color=AGE_COLORS[2], fontsize=11, va="top", ha="right", transform=T)
    ax.text(0.20, 0.9335, "road colour = age of the road", color="#d5dcec", fontsize=12, va="top", transform=T)
    ax.text(0.20, 0.916, "road width = calls merged into it", color="#d5dcec", fontsize=12, va="top", transform=T)
    ax.text(0.54, 0.9335, "●", color=DEATH, fontsize=15, va="top", transform=T)
    ax.text(0.565, 0.9335, "died", color="#d5dcec", fontsize=12, va="top", transform=T)
    ax.text(0.54, 0.916, "●", color=BIRTH, fontsize=15, va="top", transform=T)
    ax.text(0.565, 0.916, "born", color="#d5dcec", fontsize=12, va="top", transform=T)
    ax.text(0.02, 0.884, "Circles are files, dots are functions (brighter = more callers). Only what is alive that snapshot is shown.",
            color="#9aa6bf", fontsize=10.5, va="top", transform=T)
    for d, c in P["dcen"].items():
        n_here = sum(1 for f in P["groups"][d] if f in live_files)
        if n_here >= 4:
            ax.text(c[0], c[1], d, color="#9aa6bf", fontsize=9, ha="center", va="center", alpha=0.75, zorder=5, path_effects=halo)
    for f in P["hubs"]:
        if f in live_files:
            ax.text(fx[f][0], fx[f][1] - fr[f] - ext * 0.008, lay.files[f].split("/")[-1], color="#f6ecd2", fontsize=8.5,
                    ha="center", va="top", zorder=5, path_effects=halo)
    for s0 in range(max(0, s - 2), s + 1):
        age = (s - s0) + t
        if age < 3:
            fa = 1 - age / 3
            for d, k, files in sc.new_dirs[s0]:
                for f in files:
                    ax.add_patch(plt.Circle(fx[f], fr[f] + 1.2, fill=False, ec=_rgba("#ffffff", 0.3 + 0.6 * fa), lw=1.0 + 1.6 * fa, zorder=6))
    ax.text(0.5, 0.075, "\n".join(textwrap.wrap(sc.captions[s], 78)), color="#ffffff", fontsize=14, ha="center", va="top",
            linespacing=1.25, weight="bold", transform=T, path_effects=out)
    ax.text(0.02, 0.008, "A file that moves shows as a death plus a birth.", color="#7f8aa3", fontsize=9.5, va="bottom", transform=T)
    fig.savefig(path, facecolor=BG)
    return path


def render(sc: Scene, out_dir: Path, name: str = "roads", title: str | None = None, fps: float = 6,
           sub: int = 4, width: int = 720, formats=("gif", "mp4"), jobs: int = 3, progress=print) -> list[Path]:
    global _S
    _S = sc
    _CACHE.clear()
    title = title or f"Roads of {sc.ds.title}"
    tmp = Path(tempfile.mkdtemp(prefix="repo_roads_frames_"))
    jobs_list, k = [], 0
    for s in range(sc.n):
        for u in range(sub):
            jobs_list.append((s, u / sub, str(tmp / f"f{k:05d}.png"), title))
            k += 1
    progress(f"  rendering {len(jobs_list)} frames")
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = None
    if ctx and jobs > 1 and len(jobs_list) > 8:
        with ctx.Pool(jobs) as pool:
            for _ in pool.imap(render_frame, jobs_list, chunksize=4):
                pass
    else:
        for j in jobs_list:
            render_frame(j)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    ff = shutil.which("ffmpeg")
    pat = str(tmp / "f%05d.png")
    if "mp4" in formats:
        if ff:
            dst = out_dir / f"{name}.mp4"
            subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pat, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "20", str(dst)], check=True)
            made.append(dst)
        else:
            progress("  ffmpeg not found: skipping MP4 (install ffmpeg to enable it)")
    if "gif" in formats:
        dst = out_dir / f"{name}.gif"
        if ff:
            pal = tmp / "pal.png"
            subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pat, "-vf",
                            f"scale={width}:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff", str(pal)], check=True)
            subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pat, "-i", str(pal), "-lavfi",
                            f"scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                            str(dst)], check=True)
        else:
            from PIL import Image
            frames = []
            for j in jobs_list:
                im = Image.open(j[2]).convert("RGB")
                im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
                frames.append(im.quantize(colors=128, method=Image.MEDIANCUT))
            frames[0].save(dst, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=False)
        made.append(dst)
    shutil.rmtree(tmp, ignore_errors=True)
    return made
