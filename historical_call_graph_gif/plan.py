# Copyright 2026 Joe Esquibel
# SPDX-License-Identifier: Apache-2.0
"""Decide *how many* snapshots to take and *when*. Pure functions on commit times: no git, no I/O."""
from __future__ import annotations

import calendar
import datetime as dt
import re

MAX_SNAPSHOTS = 150


def _utc(t: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc)


def auto_count(span_years: float, active_quarters: int, n_commits: int, n_files: int) -> tuple[int, str]:
    """A sensible number of snapshots for this repository.

    * about one per quarter for the first 6 years, then a snapshot every ~2.5 quarters, so a 26-year
      history gets ~56 and a 3-year one gets 12;
    * never fewer than 6 (a story needs a few frames) nor more than there are commits;
    * never more than there are quarters that actually had commits: idle stretches only repeat frames;
    * capped by repository size, because every snapshot is a full scan: 48 for small repositories,
      32 for mid-sized ones, 20 for big ones.
    """
    quarters = span_years * 4
    n = quarters if quarters <= 24 else 24 + (quarters - 24) * 0.4
    n = max(6, round(n))
    cap = 48 if n_files < 1500 else 32 if n_files < 5000 else 20
    n = min(n, cap, max(active_quarters, 6), n_commits)
    n = max(n, min(2, n_commits))
    return n, (f"auto: {n} snapshots ({span_years:.1f} years, {active_quarters} quarters with commits, "
               f"~{n_files:,} files)")


def parse_every(text: str) -> tuple[str, int]:
    """'3m' -> ('months', 3); '1y' -> ('months', 12); '90d' -> ('days', 90); '2w' -> ('days', 14)."""
    m = re.fullmatch(r"\s*(\d+)\s*([dwmy])\s*", text.lower())
    if not m or int(m.group(1)) < 1:
        raise ValueError(f"cannot read --every {text!r}; use e.g. 90d, 2w, 3m or 1y")
    k, unit = int(m.group(1)), m.group(2)
    return {"d": ("days", k), "w": ("days", 7 * k), "m": ("months", k), "y": ("months", 12 * k)}[unit]


def _add_months(d: dt.datetime, k: int) -> dt.datetime:
    y, mo = d.year + (d.month - 1 + k) // 12, (d.month - 1 + k) % 12 + 1
    return d.replace(year=y, month=mo, day=min(d.day, calendar.monthrange(y, mo)[1]))


def every_targets(t0: float, t1: float, every: str) -> list[float]:
    unit, k = parse_every(every)
    start = _utc(t0)
    out, i = [], 0
    while True:
        d = _add_months(start, k * i) if unit == "months" else start + dt.timedelta(days=k * i)
        t = d.timestamp()
        if t >= t1:
            break
        out.append(t)
        i += 1
        if len(out) > MAX_SNAPSHOTS:
            raise RuntimeError(f"--every {every} would make more than {MAX_SNAPSHOTS} snapshots; "
                               "use a longer interval or --snapshots")
    out.append(t1)
    return out


def _at_or_before(times: list, targets: list) -> list:
    idx, j = [], 0
    for tt in targets:
        while j + 1 < len(times) and times[j + 1] <= tt:
            j += 1
        idx.append(j)
    return idx


def choose_indices(times: list, n="auto", every: str | None = None, spacing: str = "time",
                   n_files: int = 0) -> tuple[list, str]:
    """Indices into ``times`` (non-decreasing unix seconds, oldest first) to snapshot, and a note."""
    m = len(times)
    if m == 0:
        raise RuntimeError("no commits found")
    t0, t1 = times[0], times[-1]
    span_days = (t1 - t0) / 86400
    if every:
        idx = _at_or_before(times, every_targets(t0, t1, every))
        note = f"every {every}: {len(set(idx))} snapshots"
    else:
        if n in (None, "auto"):
            quarters = len({(d.year, (d.month - 1) // 3) for d in map(_utc, times)})
            n_eff, note = auto_count(span_days / 365.25, quarters, m, n_files)
        else:
            n_eff = max(1, min(int(n), m))
            note = f"{n_eff} snapshots"
        if n_eff == 1:
            idx = [m - 1]
        elif spacing == "time" and span_days >= 3:
            idx = _at_or_before(times, [t0 + (t1 - t0) * i / (n_eff - 1) for i in range(n_eff)])
        else:                                   # by activity (or a history too short to space by time)
            idx = [round(i * (m - 1) / (n_eff - 1)) for i in range(n_eff)]
            note += " (spaced by commit count)"
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(i)
    if len(out) < len(idx) and not every:
        note += f"; {len(out)} distinct commits after merging idle stretches"
    return out, note


def make_labels(times: list) -> list[str]:
    """Short unique labels: quarters when the spacing allows, else months, else full dates."""
    ds = [_utc(t) for t in times]
    styles = [lambda d: f"{d.year} Q{(d.month - 1) // 3 + 1}", lambda d: d.strftime("%Y-%m"),
              lambda d: d.strftime("%Y-%m-%d")]
    if len(times) < 2:
        return [styles[2](d) for d in ds]
    gaps = sorted(b - a for a, b in zip(times, times[1:]))
    med = gaps[len(gaps) // 2] / 86400
    start = 0 if med >= 75 else 1 if med >= 20 else 2
    for st in styles[start:]:                  # finest label only if a coarser one would repeat
        labels = [st(d) for d in ds]
        if len(set(labels)) == len(labels):
            return labels
    return [styles[2](d) for d in ds]
