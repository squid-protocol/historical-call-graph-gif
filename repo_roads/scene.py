"""Everything a renderer needs: dots, roads (per-snapshot call counts), captions."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .collect import Dataset
from .layout import Layout, dir_of, route

AGE_FULL = 40          # a road this many snapshots old (or older) gets the "old" colour; scaled in renderers


@dataclass
class Road:
    a: int
    b: int
    counts: list            # calls merged into this road, per snapshot (0 = absent)
    poly: np.ndarray        # (m, 2)


@dataclass
class Scene:
    ds: Dataset
    lay: Layout
    roads: list = field(default_factory=list)
    captions: list = field(default_factory=list)
    new_dirs: list = field(default_factory=list)        # per snapshot: [(dir, n_funcs, [file ids])]
    stats: list = field(default_factory=list)           # per snapshot: dict(funcs, calls, roads, born, died)

    @property
    def n(self) -> int:
        return len(self.ds.snaps)


def runs(flags) -> list:
    """Flat [start, end, start, end, ...] inclusive runs of True values."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            out += [start, i - 1]
            start = None
    if start is not None:
        out += [start, len(flags) - 1]
    return out


def road_ages(counts: list) -> list:
    """Consecutive snapshots each road has existed (0 where absent)."""
    age, out = 0, []
    for c in counts:
        age = age + 1 if c else 0
        out.append(age)
    return out


def build_scene(ds: Dataset, lay: Layout) -> Scene:
    sc = Scene(ds, lay)
    n = sc.n
    ff = lay.func_file
    pair_counts = defaultdict(lambda: [0] * n)
    for s in range(n):
        for a, b in ds.calls[s]:
            fa, fb = int(ff[a]), int(ff[b])
            if fa >= 0 and fb >= 0 and fa != fb:
                pair_counts[(min(fa, fb), max(fa, fb))][s] += 1
    for (fa, fb), counts in sorted(pair_counts.items()):
        sc.roads.append(Road(fa, fb, counts, route(lay, fa, fb)))

    seen_dirs = set()
    for s in range(n):
        alive = {i for i in ds.alive[s] if ff[i] >= 0}
        prev = {i for i in ds.alive[s - 1] if ff[i] >= 0} if s else set()
        born, died = alive - prev, prev - alive
        per_dir, per_dir_files = Counter(), defaultdict(set)
        for i in alive:
            d = dir_of(ds.funcs[i].path)
            per_dir[d] += 1
            per_dir_files[d].add(int(ff[i]))
        fresh = []
        for d, k in per_dir.most_common():
            if d not in seen_dirs and k >= 15 and s > 0:
                fresh.append((d, k, sorted(per_dir_files[d])))
        seen_dirs |= set(per_dir)
        sc.new_dirs.append(fresh)
        n_roads = sum(1 for r in sc.roads if r.counts[s])
        sc.stats.append(dict(funcs=len(alive), calls=len(ds.calls[s]), roads=n_roads,
                             born=len(born), died=len(died)))
        label = ds.snaps[s].label
        if s == 0:
            cap = f"{label}: first snapshot. {len(alive):,} functions in {len(set(int(ff[i]) for i in alive)):,} files."
        else:
            cap = f"{ds.snaps[s - 1].label} to {label}: {len(born):,} born, {len(died):,} died."
            if len(died) >= 10:
                bn = {ds.funcs[i].name for i in born}
                moved = sum(1 for i in died if ds.funcs[i].name in bn)
                pct = round(100 * moved / len(died))
                cap += (f" {pct}% of the dead reappear under the same name, so mostly moves."
                        if pct >= 50 else f" Only {pct}% of the dead reappear by name, so mostly real removals.")
        if fresh:
            cap += " New folder: " + ", ".join(f"{d} ({k:,} functions)" for d, k, _ in fresh[:3]) + "."
        sc.captions.append(cap)
    return sc
