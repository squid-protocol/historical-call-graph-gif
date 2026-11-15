# Copyright 2026 Joe Esquibel
# SPDX-License-Identifier: Apache-2.0
"""Pick N snapshots of a repository's history and extract them read-only.

Nothing here touches the user's working tree or branches: commits are listed
with ``git log`` and extracted with ``git archive`` into a scratch directory.
"""
from __future__ import annotations

import subprocess
import tarfile
import io
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


def list_commits(repo: Path, ref: str = "HEAD") -> list[tuple[int, str]]:
    """(unix time, sha) of the first-parent history of ``ref``, oldest first."""
    out = _git(repo, "log", "--first-parent", "--format=%ct %H", ref)
    rows = []
    for line in out.splitlines():
        t, sha = line.split()
        rows.append((int(t), sha))
    rows.sort()
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
    with tarfile.open(fileobj=io.BytesIO(res.stdout)) as tf:
        try:
            tf.extractall(dest, filter="data")  # py>=3.12 safe extraction
        except TypeError:
            tf.extractall(dest)
