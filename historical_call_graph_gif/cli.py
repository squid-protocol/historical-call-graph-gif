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

from . import __version__


def _say(msg: str) -> None:
    print(msg, flush=True)


def _build_outputs(ds, args, out: Path, name: str) -> None:
    from . import gif, html, layout, scene
    _say(f"laying out {len(ds.funcs):,} functions ...")
    lay = layout.build(ds, max_files=args.max_files)
    if lay.dropped_files:
        _say(f"  note: {lay.dropped_files} small files left out (limit --max-files {args.max_files})")
    sc = scene.build_scene(ds, lay)
    _say(f"  {len(sc.roads):,} roads over {sc.n} snapshots")
    formats = {f.strip() for f in args.formats.split(",") if f.strip()}
    made = []
    if "html" in formats:
        made.append(html.write_html(sc, out / f"{name}.html"))
    if formats & {"gif", "mp4"}:
        made += gif.render(sc, out, name=name, title=args.title, fps=args.fps, formats=formats,
                           jobs=args.jobs, progress=_say)
    _say("")
    for p in made:
        _say(f"  wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    if any(p.suffix == ".html" for p in made):
        _say(f"\nopen it:  historical-call-graph-gif serve {out}     (or open the .html file directly in a browser)")


def cmd_make(args) -> int:
    from . import collect, scan, snapshots
    t0 = time.time()
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
    picks = snapshots.pick_snapshots(repo, args.snapshots, ref=args.ref, since=args.since, until=args.until)
    _say(f"{title}: {len(picks)} snapshots, {picks[0].label} to {picks[-1].label}")
    dbs = scan.scan_all(repo, picks, work, jobs=args.jobs, history=args.history, progress=_say)
    ds = collect.from_snapshot_dbs(title, picks, dbs)
    _build_outputs(ds, args, out, args.basename or title)
    if not args.keep_scans:
        shutil.rmtree(work, ignore_errors=True)
    _say(f"done in {time.time() - t0:.0f}s")
    return 0


def cmd_from_db(args) -> int:
    from . import collect
    commits = collect.master_db_commits(Path(args.db), args.snapshots, args.repo)
    ds = collect.from_master_db(args.name or args.repo or Path(args.db).stem, Path(args.db), commits)
    out = Path(args.out).resolve()
    _build_outputs(ds, args, out, args.basename or ds.title)
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


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-n", "--snapshots", type=int, default=24, help="number of snapshots across the history (default 24)")
    p.add_argument("-o", "--out", default="roads-out", help="output directory (default ./roads-out)")
    p.add_argument("--formats", default="gif,mp4,html", help="comma list of gif,mp4,html (default all)")
    p.add_argument("--name", help="title shown in the outputs (default: repo name)")
    p.add_argument("--basename", help="output file name without extension")
    p.add_argument("--title", help="title text drawn on the GIF/MP4 (default 'Roads of <name>')")
    p.add_argument("--fps", type=float, default=6.0, help="GIF/MP4 frames per second (default 6)")
    p.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)), help="parallel workers")
    p.add_argument("--max-files", type=int, default=1500, help="most files drawn; busiest kept (default 1500)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="historical-call-graph-gif", description="Animated road map of a repository's call network.")
    ap.add_argument("--version", action="version", version=f"historical-call-graph-gif {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="scan N snapshots of a git repo and build the outputs")
    m.add_argument("repo", help="path to a local git repository, or a clone URL")
    m.add_argument("--ref", default="HEAD", help="branch or ref to follow (default HEAD)")
    m.add_argument("--since", help="only history after this date (e.g. 2022-01-01)")
    m.add_argument("--until", help="only history before this date")
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
