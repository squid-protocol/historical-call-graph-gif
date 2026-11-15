"""Fixed layout for a whole history: file circles, function dots, road network.

One layout is computed for every function that ever existed, so nothing moves
between frames and hubs stay put while the code around them turns over.

1. functions are packed into a circle per file;
2. file circles are placed by a force layout where files that call each other
   attract (heavily connected files end up central) and folders loosely cohere;
3. a road network of junctions (clusters of nearby files) and express lines is
   built over the files; each file-to-file road takes the cheapest path through
   it, so neighbours get direct streets and only long trips use highways.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .collect import Dataset


def dir_of(path: str, depth: int = 2) -> str:
    parts = path.split("/")[:-1]
    return "/".join(parts[:depth]) if parts else "."


def pack(radii, gap: float = 0.0):
    """Greedy spiral circle packing. Returns centres (n, 2) and the enclosing radius."""
    n = len(radii)
    R = np.asarray(radii, float) + gap
    P = np.zeros((n, 2))
    if n == 0:
        return P, 0.0
    order = np.argsort(-R, kind="stable")
    placed = [int(order[0])]
    for i in order[1:]:
        pi = np.array(placed)
        w = R[pi] ** 2
        cen = (P[pi] * w[:, None]).sum(0) / w.sum()
        ang, rad = 0.0, 0.0
        step = max(R[i] * 0.35, 0.3)
        while True:
            c = cen + rad * np.array([math.cos(ang), math.sin(ang)])
            d = np.hypot(*(P[pi] - c).T)
            if (d >= R[pi] + R[i] - 1e-9).all():
                break
            ang += step / max(rad, step)
            rad += step * 0.5 / (2 * math.pi)
        P[i] = c
        placed.append(int(i))
    w = R ** 2
    P -= (P * w[:, None]).sum(0) / w.sum()
    return P, float((np.hypot(*P.T) + R).max())


@dataclass
class Layout:
    files: list                 # file paths (index = file id)
    file_xy: np.ndarray
    file_r: np.ndarray
    func_file: np.ndarray       # func idx -> file id (-1 = dropped)
    func_xy: np.ndarray
    func_r: np.ndarray
    extent: float
    anchors: np.ndarray
    road_graph: tuple           # (dist, pred) over anchors
    file_anchor: np.ndarray     # file id -> local anchor id
    dropped_files: int = 0


def build(ds: Dataset, max_files: int = 1500, seed: int = 1, iterations: int = 450) -> Layout:
    rng = np.random.default_rng(seed)
    # ---- files that ever held a function; keep the busiest if there are too many
    per_file = defaultdict(list)
    for i, f in enumerate(ds.funcs):
        per_file[f.path].append(i)
    paths = sorted(per_file, key=lambda p: (-len(per_file[p]), p))
    dropped = max(0, len(paths) - max_files)
    paths = sorted(paths[:max_files])
    fid = {p: i for i, p in enumerate(paths)}
    nF, nfun = len(paths), len(ds.funcs)

    maxloc = np.zeros(nfun)
    for s in range(len(ds.snaps)):
        pass
    for i, f in enumerate(ds.funcs):
        maxloc[i] = max(f.info.get("loc", 1), 1)
    func_r = 0.5 + 0.12 * np.sqrt(maxloc)

    # ---- pack functions inside each file
    func_file = np.full(nfun, -1, int)
    func_off = np.zeros((nfun, 2))
    file_r = np.zeros(nF)
    for p, i in fid.items():
        idxs = sorted(per_file[p], key=lambda k: -func_r[k])
        P, enc = pack([func_r[k] for k in idxs], gap=0.25)
        file_r[i] = enc + 0.6
        for k, pt in zip(idxs, P):
            func_file[k] = i
            func_off[k] = pt

    # ---- initial file positions: pack by folder so folders start coherent
    dirs = defaultdict(list)
    for p in paths:
        dirs[dir_of(p)].append(fid[p])
    dcen, drad, fpos = {}, {}, np.zeros((nF, 2))
    for d, fs in dirs.items():
        fs = sorted(fs, key=lambda i: -file_r[i])
        P, enc = pack([file_r[i] for i in fs], gap=0.5)
        drad[d] = enc + 1.2
        for i, pt in zip(fs, P):
            fpos[i] = pt
    ds_sorted = sorted(dirs, key=lambda d: -drad[d])
    P, _ = pack([drad[d] for d in ds_sorted], gap=2.0)
    for d, pt in zip(ds_sorted, P):
        for i in dirs[d]:
            fpos[i] += pt
    P0 = fpos.copy()

    # ---- file-to-file call weights over the whole history
    ever = set()
    for s in range(len(ds.snaps)):
        ever |= ds.calls[s]
    W = Counter()
    for a, b in ever:
        fa, fb = func_file[a], func_file[b]
        if fa >= 0 and fb >= 0 and fa != fb:
            W[(min(fa, fb), max(fa, fb))] += 1
    ei = np.array([k[0] for k in W], int)
    ej = np.array([k[1] for k in W], int)
    ew = np.array([math.log1p(v) for v in W.values()], float)
    dir_idx = np.array([list(dirs).index(dir_of(p)) for p in paths]) if nF else np.zeros(0, int)
    deg = np.bincount(np.r_[ei, ej], weights=np.r_[ew, ew], minlength=nF) if len(ei) else np.zeros(nF)

    P = P0 * 0.6 + rng.normal(0, 2, P0.shape)
    for it in range(iterations if nF > 1 else 0):
        T = 1.0 - it / iterations
        dlt = P[:, None, :] - P[None, :, :]
        dist = np.sqrt((dlt ** 2).sum(-1)) + 1e-6
        gap = file_r[:, None] + file_r[None, :] + 2.0
        rep = np.where(dist < gap * 1.25, (gap * 1.25 - dist) * 0.8, 0.0)
        np.fill_diagonal(rep, 0)
        Fv = (dlt / dist[..., None] * rep[..., None]).sum(1)
        if len(ei):
            d = P[ej] - P[ei]
            l = np.sqrt((d ** 2).sum(1)) + 1e-6
            f = (d / l[:, None]) * (0.045 * ew * np.maximum(l - (file_r[ei] + file_r[ej]), 0))[:, None]
            np.add.at(Fv, ei, f)
            np.add.at(Fv, ej, -f)
        for k in range(len(dirs)):
            m = dir_idx == k
            if m.sum() > 1:
                Fv[m] += (P[m].mean(0) - P[m]) * 0.03
        Fv += -P * (0.02 + 0.002 * deg[:, None])
        P += np.clip(Fv, -8, 8) * (0.6 * T + 0.05)
    for _ in range(400 if nF > 1 else 0):          # final overlap resolution
        dlt = P[:, None, :] - P[None, :, :]
        dist = np.sqrt((dlt ** 2).sum(-1)) + 1e-6
        gap = file_r[:, None] + file_r[None, :] + 1.0
        ov = np.where(dist < gap, (gap - dist) / 2, 0)
        np.fill_diagonal(ov, 0)
        if ov.max() < 0.05:
            break
        P += (dlt / dist[..., None] * ov[..., None]).sum(1) * 0.5
    if nF:
        P -= P.mean(0)
    func_xy = np.zeros((nfun, 2))
    ok = func_file >= 0
    func_xy[ok] = P[func_file[ok]] + func_off[ok]
    extent = float((np.hypot(*P.T) + file_r).max() + 5) if nF else 10.0

    anchors, graph, file_anchor = _road_network(P, file_r, W, nF)
    return Layout(paths, P, file_r, func_file, func_xy, func_r, extent, anchors, graph, file_anchor, dropped)


def _road_network(P, file_r, W, nF):
    """Junctions (local clusters), regions, express lines; all-pairs cheapest paths."""
    if nF < 8:
        return np.zeros((0, 2)), (np.zeros((0, 0)), np.zeros((0, 0), int)), np.zeros(nF, int)
    from scipy.cluster.vq import kmeans2
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import shortest_path

    def km(X, k, seed):
        k = max(1, min(k, len(X)))
        c, lab = kmeans2(X, k, minit="++", seed=seed)
        keep = sorted(set(lab.tolist()))
        remap = {o: i for i, o in enumerate(keep)}
        return c[keep], np.array([remap[l] for l in lab.tolist()])

    k_loc = int(min(48, max(4, round(nF / 20))))
    k_reg = int(min(8, max(2, round(k_loc / 6))))
    LOCC, LOCLAB = km(P, k_loc, 3)
    REGC, REGLAB = km(LOCC, k_reg, 5)
    NL, NR = len(LOCC), len(REGC)
    ANCH = np.vstack([LOCC, REGC])
    G = lil_matrix((NL + NR, NL + NR))

    def add(i, j, w):
        L = float(np.hypot(*(ANCH[i] - ANCH[j]))) * w
        if G[i, j] == 0 or L < G[i, j]:
            G[i, j] = L
            G[j, i] = L

    def knn(X, k):
        d = np.hypot(*(X[:, None, :] - X[None, :, :]).transpose(2, 0, 1))
        out = set()
        for i in range(len(X)):
            for j in np.argsort(d[i])[1:k + 1]:
                out.add((min(i, int(j)), max(i, int(j))))
        return out
    for i, j in knn(LOCC, 4):
        add(i, j, 1.0)                                # streets
    for i in range(NL):
        add(i, NL + int(REGLAB[i]), 1.0)              # junction -> region
    for i, j in knn(REGC, min(3, NR - 1) if NR > 1 else 0):
        add(NL + i, NL + j, 0.5)                      # highways between regions (discounted)
    tr = Counter()
    for (a, b), v in W.items():
        la, lb = int(LOCLAB[a]), int(LOCLAB[b])
        if la != lb:
            tr[(min(la, lb), max(la, lb))] += v
    for (i, j), _ in tr.most_common(40):
        add(i, j, 0.6)                                # express lines for the heaviest pairs
    dist, pred = shortest_path(G.tocsr(), directed=False, return_predecessors=True)
    return ANCH, (dist, pred), LOCLAB.astype(int)


def route(lay: Layout, fa: int, fb: int, beta: float = 0.96, samples: int = 8) -> np.ndarray:
    """Smooth polyline for the road between two files, following the road network."""
    A, B = lay.file_xy[fa], lay.file_xy[fb]
    if len(lay.anchors) == 0:
        mid = (A + B) / 2
        pts = [A, mid, B]
    else:
        la, lb = int(lay.file_anchor[fa]), int(lay.file_anchor[fb])
        if la == lb:
            pts = [A, lay.anchors[la], B]
        else:
            pred = lay.road_graph[1]
            path = [lb]
            while path[-1] != la:
                p = pred[la, path[-1]]
                if p < 0:
                    path = [la, lb]
                    break
                path.append(int(p))
            path = path[::-1]
            pts = [A] + [lay.anchors[i] for i in path] + [B]
    P = np.array(pts, float)
    n = len(P)
    line = P[0] + (P[-1] - P[0]) * np.linspace(0, 1, n)[:, None]
    P = beta * P + (1 - beta) * line
    P = np.vstack([P[0], P[0], P, P[-1], P[-1]])
    t = np.linspace(0, 1, samples + 1)[:-1]
    out = []
    for i in range(len(P) - 3):
        a, b, c, d = P[i], P[i + 1], P[i + 2], P[i + 3]
        out.append(((-a + 3 * b - 3 * c + d)[None] * t[:, None] ** 3 + (3 * a - 6 * b + 3 * c)[None] * t[:, None] ** 2
                    + (-3 * a + 3 * c)[None] * t[:, None] + (a + 4 * b + c)[None]) / 6)
    out.append(((P[-3] + 4 * P[-2] + P[-1]) / 6)[None])
    return np.vstack(out)
