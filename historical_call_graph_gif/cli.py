# Copyright 2026 Joe Esquibel
# SPDX-License-Identifier: Apache-2.0
"""historical-call-graph-gif command line."""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import shutil
import socketserver
import subprocess
import sys
import time
from pathlib import Path

from . import __version__, style as style_mod


def _say(msg: str) -> None:
    print(msg, flush=True)


def _make_style(args):
    try:
        return style_mod.resolve(
            preset=args.preset, size=args.size, title=args.title, subtitle=args.subtitle, credit=args.credit,
            show_legend=not args.no_legend, show_captions=not args.no_captions, fit=args.fit, zoom=args.zoom,
            focus=args.focus, quality=args.quality, fps=args.fps, duration=args.duration, hold=args.hold,
            frames_per_snapshot=args.frames_per_snapshot, smooth=not args.no_smooth, gif_width=args.gif_width,
            text_scale=args.text_scale, formats=args.formats)
    except ValueError as e:
        raise RuntimeError(str(e)) from None


def _build_outputs(ds, args, out: Path, name: str, st) -> None:
    from . import gif, html, layout, scene
    _say(f"laying out {len(ds.funcs):,} functions ...")
    lay = layout.build(ds, max_files=args.max_files)
    if lay.dropped_files:
        _say(f"  note: {lay.dropped_files} files ({lay.dropped_funcs:,} functions) are not drawn (limit --max-files {args.max_files}); raise it to draw more")
    sc = scene.build_scene(ds, lay)
    _say(f"  {len(sc.roads):,} roads over {sc.n} snapshots")
    formats = {f.strip() for f in st.formats.split(",") if f.strip()}
    unknown = formats - {"gif", "mp4", "html", "png", "frames"}
    if unknown:
        raise RuntimeError(f"unknown format(s): {', '.join(sorted(unknown))}; use gif, mp4, html, png, frames")
    made = []
    if "html" in formats:
        made.append(html.write_html(sc, out / f"{name}.html"))
    if formats & {"gif", "mp4", "png", "frames"}:
        made += gif.render(sc, out, name=name, style=st, jobs=args.jobs, progress=_say)
    _say("")
    for p in made:
        _say(f"  wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    if any(p.suffix == ".html" for p in made):
        _say(f"\nopen it:  historical-call-graph-gif serve {out}     (or open the .html file directly in a browser)")


def cmd_make(args) -> int:
    from . import collect, scan, snapshots
    t0 = time.time()
    st = _make_style(args)                    # fail on a bad size/preset now, not after the scans
    target = args.repo
    out = Path(args.out).resolve()
    work = out / ".work"
    out.mkdir(parents=True, exist_ok=True)
    if Path(target).exists():
        repo = Path(target).resolve()
        title = args.name or repo.name
    else:
        repo = work / "clone"
        title = args.name or target.rstrip("/").split("/")[-1].removesuffix(".git")
        if not repo.exists():
            _say(f"cloning {target} ...")
            work.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--bare", "--quiet", target, str(repo)], check=True)
    if not (repo / ".git").exists() and not (repo / "HEAD").exists():
        _say(f"error: {repo} is not a git repository")
        return 2
    picks, note = snapshots.plan_snapshots(repo, args.snapshots, ref=args.ref, since=args.since, until=args.until,
                                           every=args.every, spacing=args.spacing)
    _say(f"{title}: {len(picks)} snapshots, {picks[0].label} to {picks[-1].label}   [{note}]")
    picks, dbs = scan.scan_all(repo, picks, work, jobs=args.jobs, history=args.history, progress=_say)
    ds = collect.drop_leading_empty(collect.from_snapshot_dbs(title, picks, dbs))
    _build_outputs(ds, args, out, args.basename or title, st)
    if not args.keep_scans:
        shutil.rmtree(work, ignore_errors=True)
    _say(f"done in {time.time() - t0:.0f}s")
    return 0


def cmd_from_db(args) -> int:
    from . import collect
    st = _make_style(args)
    commits = collect.master_db_commits(Path(args.db), args.snapshots, args.repo)
    ds = collect.drop_leading_empty(collect.from_master_db(args.name or args.repo or Path(args.db).stem, Path(args.db), commits))
    out = Path(args.out).resolve()
    _build_outputs(ds, args, out, args.basename or ds.title, st)
    return 0


def cmd_serve(args) -> int:
    root = Path(args.dir).resolve()
    if not root.is_dir():
        _say(f"error: {root} is not a directory")
        return 2
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        htmls = sorted(p.name for p in root.glob("*.html"))
        _say(f"serving {root} at http://{args.host}:{args.port}/   (Ctrl-C to stop)")
        for h in htmls:
            _say(f"  http://{args.host}:{args.port}/{h}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            _say("\nstopped")
    return 0


def _count(text: str):
    if text.lower() == "auto":
        return "auto"
    try:
        n = int(text)
        if n >= 1:
            return n
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("use a whole number or 'auto'")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-n", "--snapshots", type=_count, default="auto",
                   help="number of snapshots, or 'auto' (default): sized from the history length, activity and repo size")
    p.add_argument("-o", "--out", default="roads-out", help="output directory (default ./roads-out)")
    g = p.add_argument_group("output")
    g.add_argument("--formats", help="comma list of gif,mp4,html,png,frames (default: depends on --preset; 'shorts' makes an MP4)")
    g.add_argument("--preset", choices=style_mod.preset_names(), default="default",
                   help="frame shape: landscape 1920x1080, square 1080x1080, portrait 1080x1350, "
                        "shorts 1080x1920 for phones (YouTube Shorts, Reels, TikTok); aliases: phone, vertical, reels, tiktok")
    g.add_argument("--size", metavar="WxH", help="exact size in pixels, e.g. 1080x1920 (overrides the preset's size)")
    g.add_argument("--quality", choices=list(style_mod.QUALITY), default="standard",
                   help="encoder quality: draft (small, fast) .. max (largest); default standard")
    g.add_argument("--basename", help="output file name without extension")
    g.add_argument("--gif-width", type=int, help="GIF width in pixels (default depends on --quality)")
    t = p.add_argument_group("titles and text")
    t.add_argument("--name", help="project name used in titles and file names (default: repo name)")
    t.add_argument("--title", help="title drawn on the GIF/MP4/PNG (default 'Roads of <name>')")
    t.add_argument("--subtitle", help="replaces the one-line explanation under the legend")
    t.add_argument("--credit", help="small text in the bottom-right corner, e.g. a channel handle")
    t.add_argument("--no-captions", action="store_true", help="hide the narration captions at the bottom")
    t.add_argument("--no-legend", action="store_true", help="hide the legend and explanation line")
    t.add_argument("--text-scale", type=float, help="make all text bigger or smaller (the shorts preset uses 1.45)")
    v = p.add_argument_group("framing")
    v.add_argument("--fit", choices=["contain", "fill"], default="contain", help="contain: whole map visible (default); fill: crop to fill the frame")
    v.add_argument("--zoom", type=float, default=1.0, help="zoom in (>1) or out (<1) from the fitted view")
    v.add_argument("--focus", metavar="PATH", help="frame one folder or file prefix, e.g. lib/ or src/network")
    m_ = p.add_argument_group("timing")
    m_.add_argument("--duration", type=float, help="seconds of animation (the shorts preset uses 45)")
    m_.add_argument("--fps", type=float, help="distinct frames per second of the source; ignored when --duration is given")
    m_.add_argument("--hold", type=float, default=2.0, help="seconds to linger on the final frame (default 2; rounded up to whole source frames)")
    m_.add_argument("--frames-per-snapshot", type=int, default=4, help="rendered frames between snapshots (default 4; more = smoother, slower)")
    m_.add_argument("--no-smooth", action="store_true", help="MP4: do not dissolve between rendered frames")
    p.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)), help="parallel workers")
    p.add_argument("--max-files", type=int, default=4000, help="most files drawn; files with the most functions are kept (default 4000)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="historical-call-graph-gif", description="Animated road map of a repository's call network.")
    ap.add_argument("--version", action="version", version=f"historical-call-graph-gif {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="scan N snapshots of a git repo and build the outputs")
    m.add_argument("repo", help="path to a local git repository, or a clone URL")
    m.add_argument("--ref", default="HEAD", help="branch or ref to follow (default HEAD)")
    m.add_argument("--since", help="only history after this date (e.g. 2022-01-01)")
    m.add_argument("--until", help="only history before this date")
    m.add_argument("--every", metavar="INTERVAL", help="one snapshot per interval instead of a count: 90d, 2w, 3m, 1y")
    m.add_argument("--spacing", choices=["time", "activity"], default="time",
                   help="space snapshots evenly in time (default) or in commit count, which gives busy periods more frames")
    m.add_argument("--history", action="store_true", help="let GitGalaxy read git history too (slower; adds churn/stability)")
    m.add_argument("--keep-scans", action="store_true", help="keep the per-snapshot scan databases")
    _common(m)
    m.set_defaults(fn=cmd_make)
    d = sub.add_parser("from-db", help="build outputs from an existing multi-commit GitGalaxy database")
    d.add_argument("db")
    d.add_argument("--repo", help="repo_name inside the database")
    _common(d)
    d.set_defaults(fn=cmd_from_db)
    s = sub.add_parser("serve", help="self-host the generated page on localhost")
    s.add_argument("dir", nargs="?", default="roads-out")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to share on your network")
    s.set_defaults(fn=cmd_serve)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except RuntimeError as e:
        _say(f"error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
