# Copyright 2026 Joe Esquibel
# SPDX-License-Identifier: Apache-2.0
"""Run the GitGalaxy scanner (``galaxyscope``) once per snapshot."""
from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .snapshots import Pick, extract


def find_scanner() -> str:
    path = os.environ.get("GALAXYSCOPE_BIN") or shutil.which("galaxyscope")
    if not path:
        raise RuntimeError(
            "The GitGalaxy scanner 'galaxyscope' was not found. Install it with "
            "'pip install gitgalaxy' (or set GALAXYSCOPE_BIN=/path/to/galaxyscope)."
        )
    return path


def scan_snapshot(repo: Path, pick: Pick, idx: int, work: Path, scanner: str, history: bool) -> Path:
    """Extract one commit, scan it, return the path of its SQLite database."""
    name = f"snap_{idx:03d}"
    src = work / name
    db = work / f"{name}_galaxy_master.db"
    if db.exists():                      # resumable: a finished snapshot is kept
        return db
    if src.exists():
        shutil.rmtree(src)
    extract(repo, pick.commit, src)
    env = dict(os.environ)
    if not history:
        env["GITGALAXY_DISABLE_GIT_HISTORY"] = "1"   # structure only; far faster
    res = subprocess.run([scanner, str(src), "--db-only"], cwd=work, env=env,
                         capture_output=True, text=True, errors="replace")
    shutil.rmtree(src, ignore_errors=True)
    if not db.exists():
        tail = (res.stdout + res.stderr)[-600:]
        raise RuntimeError(f"scan of {pick.commit[:10]} produced no database:\n{tail}")
    return db


def scan_all(repo: Path, picks: list[Pick], work: Path, jobs: int = 3, history: bool = False,
             progress=print) -> tuple[list[Pick], list[Path]]:
    """Scan every snapshot; a snapshot that fails is skipped with a warning, not fatal.

    Returns the picks that were scanned and their databases, in order.
    """
    scanner = find_scanner()
    work.mkdir(parents=True, exist_ok=True)
    dbs: list[Path | None] = [None] * len(picks)

    def one(i: int) -> None:
        try:
            dbs[i] = scan_snapshot(repo, picks[i], i, work, scanner, history)
            progress(f"  scanned {i + 1}/{len(picks)}  {picks[i].label}  {picks[i].commit[:10]}")
        except RuntimeError as e:
            progress(f"  skipped {i + 1}/{len(picks)}  {picks[i].label}  {picks[i].commit[:10]}: {str(e).splitlines()[0]}")

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        list(ex.map(one, range(len(picks))))
    ok = [(p, d) for p, d in zip(picks, dbs) if d is not None]
    if not ok:
        raise RuntimeError("no snapshot could be scanned")
    return [p for p, _ in ok], [d for _, d in ok]
