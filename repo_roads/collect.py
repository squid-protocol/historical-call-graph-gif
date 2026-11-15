"""Read GitGalaxy scan databases into one history-wide ``Dataset``.

A function is identified across snapshots by (file path, name, nth same-name
function in that file). A function that moves to another file therefore shows
as a death plus a birth; the captions say so rather than hiding it.
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# function_data columns that are metadata, not keyword/signal counters
_BASE = {
    "id", "file_id", "parent_class_id", "func_name", "complexity", "loc", "start_line", "args",
    "usage_status", "keyword_density", "func_archetype", "func_z_score", "docstring",
    "calls_out_to", "token_mass", "is_public", "is_documented",
}
_FILE_COLS = ["file_archetype", "language", "total_loc", "risk_cognitive_load", "risk_safety_score",
              "risk_tech_debt", "risk_verification", "risk_api_exposure", "risk_concurrency",
              "risk_state_flux", "risk_dead_code", "risk_spec_match", "risk_stability",
              "risk_churn", "risk_documentation", "risk_secrets_risk"]


@dataclass
class Snapshot:
    label: str
    date: str
    commit: str


@dataclass
class FuncRec:
    path: str
    name: str
    nth: int
    first: int
    last: int
    info: dict = field(default_factory=dict)      # metrics from the last snapshot it was seen in


@dataclass
class Dataset:
    title: str
    snaps: list[Snapshot]
    funcs: list[FuncRec] = field(default_factory=list)
    alive: list[set] = field(default_factory=list)            # per snapshot: set of func idx
    calls: list[set] = field(default_factory=list)            # per snapshot: set of (src, dst)
    files: dict = field(default_factory=dict)                 # path -> info dict (latest)
    _index: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.snaps)
        if not self.alive:
            self.alive = [set() for _ in range(n)]
            self.calls = [set() for _ in range(n)]


def _cols(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def _parse_calls(v) -> list:
    if not v:
        return []
    try:
        out = json.loads(v)
        return out if isinstance(out, list) else []
    except (ValueError, TypeError):
        return []


def _ingest(ds: Dataset, s: int, file_map: dict, rows, cols: list[str]) -> None:
    """Add one snapshot's functions. ``rows`` are function_data rows in ``cols`` order."""
    c = {name: i for i, name in enumerate(cols)}
    sig_cols = [n for n in cols if n not in _BASE]
    per_file = defaultdict(list)
    for r in rows:
        fm = file_map.get(r[c["file_id"]])
        if fm is None:
            continue
        per_file[fm[0]].append(r)
    by_name_global = defaultdict(list)
    pending_calls = []
    names_in_file = defaultdict(dict)
    for path, frows in per_file.items():
        frows.sort(key=lambda r: (r[c["start_line"]] or 0) if "start_line" in c else 0)
        seen = defaultdict(int)
        for r in frows:
            name = r[c["func_name"]]
            nth = seen[name]
            seen[name] += 1
            key = (path, name, nth)
            idx = ds._index.get(key)
            if idx is None:
                idx = len(ds.funcs)
                ds._index[key] = idx
                ds.funcs.append(FuncRec(path, name, nth, s, s))
            rec = ds.funcs[idx]
            rec.last = s
            sig = {n: int(r[c[n]]) for n in sig_cols if isinstance(r[c[n]], (int, float)) and r[c[n]]}
            rec.info = {
                "loc": r[c["loc"]] or 0,
                "cx": r[c["complexity"]] or 0,
                "args": r[c["args"]] or 0,
                "arch": (r[c["func_archetype"]] if "func_archetype" in c else "") or "",
                "z": round(float(r[c["func_z_score"]] or 0.0), 2) if "func_z_score" in c else 0.0,
                "pub": int(r[c["is_public"]] or 0) if "is_public" in c else 0,
                "doc": int(r[c["is_documented"]] or 0) if "is_documented" in c else 0,
                "tok": r[c["token_mass"]] or 0 if "token_mass" in c else 0,
                "sig": sig,
            }
            ds.alive[s].add(idx)
            names_in_file[path].setdefault(name, idx)
            by_name_global[name].append(idx)
            pending_calls.append((idx, path, _parse_calls(r[c["calls_out_to"]]) if "calls_out_to" in c else []))
    edges = ds.calls[s]
    for idx, path, callees in pending_calls:
        own = ds.funcs[idx].name
        for cn in set(callees):
            if cn == own:
                continue
            tgt = names_in_file[path].get(cn)
            if tgt is None:
                cands = by_name_global.get(cn)
                if cands and len(cands) == 1:
                    tgt = cands[0]
            if tgt is not None and tgt != idx:
                edges.add((idx, tgt))


def _file_info(con: sqlite3.Connection) -> dict:
    have = _cols(con, "file_data")
    sel = [x for x in _FILE_COLS if x in have]
    out = {}
    for row in con.execute(f"SELECT file_path, {', '.join(sel)} FROM file_data"):
        out[row[0]] = dict(zip(sel, row[1:]))
    return out


def _fold_files(ds: Dataset, info: dict) -> None:
    for p, d in info.items():
        ds.files[p] = d


def from_snapshot_dbs(title: str, picks, dbs: list[Path]) -> Dataset:
    """Per-snapshot databases (what ``repo-roads make`` produces)."""
    ds = Dataset(title, [Snapshot(p.label, p.date, p.commit) for p in picks])
    for s, db in enumerate(dbs):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            file_map = {i: (p, l) for i, p, l in con.execute("SELECT id, file_path, language FROM file_data")}
            cols = _cols(con, "function_data")
            rows = con.execute(f"SELECT {', '.join(cols)} FROM function_data").fetchall()
            _ingest(ds, s, file_map, rows, cols)
            _fold_files(ds, _file_info(con))
        finally:
            con.close()
    return ds


def from_master_db(title: str, db: Path, commits: list[tuple[str, str, str]]) -> Dataset:
    """A multi-commit database (one ``repo_data`` row per commit, e.g. a temporal-crucible DB).

    ``commits``: (label, date, commit_hash) per snapshot, oldest first.
    """
    ds = Dataset(title, [Snapshot(*c) for c in commits])
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    want = {h: i for i, (_, _, h) in enumerate(commits)}
    fmaps = defaultdict(dict)
    for fid, h, p, lang in con.execute("SELECT id, commit_hash, file_path, language FROM file_data"):
        if h in want:
            fmaps[want[h]][fid] = (p, lang)
    cols = _cols(con, "function_data")
    for s in range(len(commits)):
        ids = list(fmaps[s])
        rows = []
        for k in range(0, len(ids), 400):
            chunk = ids[k:k + 400]
            rows += con.execute(
                f"SELECT {', '.join(cols)} FROM function_data WHERE file_id IN ({','.join('?' * len(chunk))})", chunk
            ).fetchall()
        _ingest(ds, s, fmaps[s], rows, cols)
    con.close()
    return ds


def master_db_commits(db: Path, n: int, repo_name: str | None = None) -> list[tuple[str, str, str]]:
    """Evenly spaced (in time) commits from a multi-commit database."""
    import datetime as dt
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    q = "SELECT commit_date, commit_hash FROM repo_data"
    args: tuple = ()
    if repo_name:
        q += " WHERE repo_name = ?"
        args = (repo_name,)
    rows = sorted(con.execute(q + " ORDER BY commit_date", args).fetchall())
    con.close()
    if not rows:
        raise RuntimeError("no commits in database")

    def t(d):
        return dt.datetime.fromisoformat(d[:10]).timestamp()
    t0, t1 = t(rows[0][0]), t(rows[-1][0])
    n = max(1, min(n, len(rows)))
    picks, j = [], 0
    for i in range(n):
        tt = t0 + (t1 - t0) * (i / (n - 1) if n > 1 else 1)
        while j + 1 < len(rows) and t(rows[j + 1][0]) <= tt:
            j += 1
        if not picks or rows[j][1] != picks[-1][1]:
            picks.append((rows[j][0][:10], rows[j][1]))
    span = (t1 - t0) / 86400
    out = []
    for d, h in picks:
        y, m = int(d[:4]), int(d[5:7])
        out.append((d if span < 730 else f"{y} Q{(m - 1) // 3 + 1}", d, h))
    return out
