# Copyright 2026 Joe Esquibel
# SPDX-License-Identifier: Apache-2.0
"""Self-contained interactive viewer: one HTML file, no network, no build step."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .layout import dir_of
from .scene import Scene, runs

RISK_SHOWN = ["risk_cognitive_load", "risk_safety_score", "risk_tech_debt", "risk_verification",
              "risk_api_exposure", "risk_concurrency", "risk_state_flux", "risk_dead_code",
              "risk_documentation", "risk_secrets_risk"]


def _r(x, nd=1):
    return round(float(x), nd)


def scene_to_dict(sc: Scene) -> dict:
    ds, lay = sc.ds, sc.lay
    n = sc.n
    keep = [i for i in range(len(ds.funcs)) if lay.func_file[i] >= 0]
    remap = {old: new for new, old in enumerate(keep)}
    funcs = []
    for old in keep:
        f = ds.funcs[old]
        flags = [old in ds.alive[s] for s in range(n)]
        info = f.info
        sig = dict(sorted(info.get("sig", {}).items(), key=lambda kv: -kv[1])[:12])
        funcs.append([f.name, int(lay.func_file[old]), _r(lay.func_xy[old][0]), _r(lay.func_xy[old][1]), _r(lay.func_r[old], 2),
                      info.get("loc", 0), info.get("cx", 0), info.get("args", 0), info.get("arch", ""), info.get("z", 0),
                      info.get("pub", 0), info.get("doc", 0), info.get("tok", 0), sig, runs(flags)])
    ever = Counter()
    edge_flags = {}
    for s in range(n):
        for a, b in ds.calls[s]:
            if a in remap and b in remap:
                edge_flags.setdefault((remap[a], remap[b]), [False] * n)[s] = True
    edges = [[a, b, runs(fl)] for (a, b), fl in edge_flags.items()]
    files = []
    for i, p in enumerate(lay.files):
        info = ds.files.get(p, {})
        files.append({"p": p, "x": _r(lay.file_xy[i][0]), "y": _r(lay.file_xy[i][1]), "r": _r(lay.file_r[i]),
                      "arch": info.get("file_archetype") or "", "lang": info.get("language") or "",
                      "loc": info.get("total_loc") or 0,
                      "risk": {k[5:]: _r(info[k]) for k in RISK_SHOWN if info.get(k) is not None}})
    roads = [{"a": r.a, "b": r.b, "c": r.counts, "p": [_r(v) for v in r.poly.reshape(-1)]} for r in sc.roads]
    wt = Counter()
    for r in sc.roads:
        w = max(r.counts)
        wt[r.a] += w
        wt[r.b] += w
    groups = {}
    for i, p in enumerate(lay.files):
        groups.setdefault(dir_of(p), []).append(i)
    dirs = [{"d": d, "x": _r(lay.file_xy[fs][:, 0].mean()), "y": _r(lay.file_xy[fs][:, 1].mean()), "f": fs} for d, fs in groups.items()]
    return {
        "meta": {"title": ds.title, "ext": _r(lay.extent), "dropped_files": lay.dropped_files,
                 "snaps": [{"label": s.label, "date": s.date, "commit": s.commit[:10], **sc.stats[k]} for k, s in enumerate(ds.snaps)],
                 "caps": sc.captions},
        "files": files, "funcs": funcs, "edges": edges, "roads": roads, "dirs": dirs,
        "hubs": [p for p, _ in wt.most_common(6)],
        "newdirs": [[{"d": d, "n": k, "f": fs} for d, k, fs in nd] for nd in sc.new_dirs],
    }


def write_html(sc: Scene, dst: Path) -> Path:
    tpl = (Path(__file__).parent / "web" / "viewer.html").read_text(encoding="utf-8")
    payload = json.dumps(scene_to_dict(sc), separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("/*__DATA__*/null", payload)
    title = sc.ds.title.replace("<", "").replace(">", "")
    html = html.replace("__TITLE__", title)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(html, encoding="utf-8")
    return dst
