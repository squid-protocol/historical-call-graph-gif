# Copyright 2026 Joe Esquibel
# SPDX-License-Identifier: Apache-2.0
"""Pick N snapshots of a repository's history and extract them read-only.

Nothing here touches the user's working tree or branches: commits are listed
with ``git log`` and extracted with ``git archive`` into a scratch directory.
"""
from __future__ import annotations

import subprocess
import io
import tarfile
from pathlib import PurePosixPath
from dataclasses import dataclass
from pathlib import Path

from .plan import choose_indices, make_labels


@dataclass
class Pick:
    commit: str
    date: str      # ISO date (committer date, UTC)
    label: str     # short label shown in the viewer


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout


def tree_is_empty(repo: Path, commit: str) -> bool:
    """True when ``commit`` has no files (some imported histories start with an empty commit)."""
    return not _git(repo, "ls-tree", "-r", "--name-only", "-z", commit).strip("\0")


def list_commits(repo: Path, ref: str = "HEAD") -> list[tuple[int, str]]:
    """(unix time, sha) of the first-parent history of ``ref``, oldest first.

    Kept in git's own order (not re-sorted by time): commits that share a timestamp, such as a root
    commit and its first child in an imported history, must stay in parent-to-child order.
    Times are made non-decreasing so that picking evenly in time is well defined.
    """
    out = _git(repo, "log", "--first-parent", "--format=%ct %H", ref)
    rows = []
    for line in reversed(out.splitlines()):
        t, sha = line.split()
        rows.append((int(t), sha))
    run = 0
    for i, (t, sha) in enumerate(rows):
        run = max(run, t)
        rows[i] = (run, sha)
    return rows


def plan_snapshots(repo: Path, n="auto", ref: str = "HEAD", since: str | None = None, until: str | None = None,
                   every: str | None = None, spacing: str = "time") -> tuple[list[Pick], str]:
    """Choose snapshot commits; returns the picks and a one-line explanation of the choice.

    ``n`` is a number or "auto" (sized from the history length, activity and repository size).
    ``every`` ("3m", "1y", "90d") takes a snapshot per interval instead. ``spacing`` is "time" (evenly in
    time, so a quiet year and a busy year get equal room) or "activity" (evenly in commit count).
    """
    commits = list_commits(repo, ref)
    if since:
        t0 = int(_git(repo, "log", "-1", "--format=%ct", f"--until={since}", ref).strip() or 0)
        commits = [c for c in commits if c[0] >= t0] or commits
    if until:
        t1 = int(_git(repo, "log", "-1", "--format=%ct", f"--until={until}", ref).strip() or 10 ** 12)
        commits = [c for c in commits if c[0] <= t1] or commits
    if not commits:
        raise RuntimeError("no commits found")
    k = 0                                   # histories imported from SVN/CVS can start with empty commits
    while k < min(len(commits) - 1, 500) and tree_is_empty(repo, commits[k][1]):
        k += 1
    commits = commits[k:]
    times = [t for t, _ in commits]
    n_files = 0
    if not every and n in (None, "auto"):
        n_files = len(_git(repo, "ls-tree", "-r", "--name-only", commits[-1][1]).splitlines())
    idx, note = choose_indices(times, n, every, spacing, n_files)
    labels = make_labels([times[i] for i in idx])
    picks = [Pick(commits[i][1], _utc_date(commits[i][0]), lab) for i, lab in zip(idx, labels)]
    return picks, note


def pick_snapshots(repo: Path, n="auto", ref: str = "HEAD", since: str | None = None,
                   until: str | None = None, every: str | None = None, spacing: str = "time") -> list[Pick]:
    return plan_snapshots(repo, n, ref, since, until, every, spacing)[0]


def _utc_date(t: int) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d")


def extract(repo: Path, commit: str, dest: Path) -> None:
    """Export the tree of ``commit`` into ``dest`` (read-only on the repository)."""
    dest.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit], capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"git archive {commit} failed: {res.stderr.decode(errors='replace')}")
    try:
        tf = tarfile.open(fileobj=io.BytesIO(res.stdout))
    except tarfile.ReadError:
        raise RuntimeError(f"commit {commit[:10]} has no files to scan") from None
    with tf:
        members = list(_safe_members(tf))
        try:
            tf.extractall(dest, members=members, filter="data")  # py>=3.12 safe extraction
        except TypeError:
            tf.extractall(dest, members=members)


def _safe_members(tf: tarfile.TarFile):
    """Regular files and directories with safe relative paths only.

    Symlinks, hardlinks and device files are skipped: the scanner reads source text, and histories
    do contain absolute or escaping symlinks that a strict extractor would abort on.
    """
    for m in tf:
        if not (m.isreg() or m.isdir()):
            continue
        parts = PurePosixPath(m.name).parts
        if m.name.startswith("/") or ".." in parts:
            continue
        yield m
