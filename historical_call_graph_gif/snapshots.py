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


def pick_snapshots(repo: Path, n: int, ref: str = "HEAD", since: str | None = None,
                   until: str | None = None) -> list[Pick]:
    """N commits evenly spaced in *time* between the first and last commit.

    Evenly spaced in time (not in commit count) so that a quiet year and a
    busy year each get the same share of the animation.
    """
    commits = list_commits(repo, ref)
    if since:
        t0 = int(_git(repo, "log", "-1", "--format=%ct", f"--until={since}", ref).strip() or 0)
        commits = [c for c in commits if c[0] >= t0] or commits
    if until:
        commits = [c for c in commits if c[0] <= int(_git(repo, "log", "-1", "--format=%ct", f"--until={until}", ref).strip() or 10**12)] or commits
    if not commits:
        raise RuntimeError("no commits found")
    k = 0                                   # histories imported from SVN/CVS can start with empty commits
    while k < min(len(commits) - 1, 500) and tree_is_empty(repo, commits[k][1]):
        k += 1
    commits = commits[k:]
    n = max(1, min(n, len(commits)))
    t_first, t_last = commits[0][0], commits[-1][0]
    picks: list[tuple[int, str]] = []
    if n == 1 or t_first == t_last:
        picks = [commits[-1]]
    else:
        times = [t_first + (t_last - t_first) * i / (n - 1) for i in range(n)]
        j = 0
        for tt in times:
            while j + 1 < len(commits) and commits[j + 1][0] <= tt:
                j += 1
            if not picks or commits[j][1] != picks[-1][1]:
                picks.append(commits[j])
    import datetime as dt
    out = []
    span_days = (t_last - t_first) / 86400
    for t, sha in picks:
        d = dt.datetime.fromtimestamp(t, dt.timezone.utc)
        label = d.strftime("%Y-%m-%d") if span_days < 2 * 365 else f"{d.year} Q{(d.month - 1) // 3 + 1}"
        out.append(Pick(sha, d.strftime("%Y-%m-%d"), label))
    return out


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
