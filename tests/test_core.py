# Copyright 2026 Joe Esquibel
# SPDX-License-Identifier: Apache-2.0
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from historical_call_graph_gif import collect, html, layout, scene, snapshots


def make_dataset(n_snaps=4):
    """Two folders; a hub in lib/core.c; functions born over time; one moves file."""
    snaps = [collect.Snapshot(f"s{i}", f"2020-0{i + 1}-01", f"c{i}" * 5) for i in range(n_snaps)]
    ds = collect.Dataset("demo", snaps)
    cols = ["id", "file_id", "func_name", "complexity", "loc", "start_line", "args", "func_archetype",
            "func_z_score", "calls_out_to", "is_public", "is_documented", "token_mass", "struct_branch"]

    def snap(s, funcs):
        fm, rows = {}, []
        for k, (path, name, callees) in enumerate(funcs):
            fid = fm.setdefault(path, (len(fm) + 1, None))[0]
            rows.append((k, fid, name, 3, 20 + k, k * 10, 2, "Compute Cores", 0.5, json.dumps(callees), 1, 0, 7, 2))
        collect._ingest(ds, s, {v[0]: (p, "c") for p, v in fm.items()}, rows, cols)

    base = [("lib/core.c", "hub", []), ("lib/core.c", "util", ["hub"]), ("src/main.c", "main", ["hub", "util"])]
    snap(0, base)
    snap(1, base + [("src/cli.c", "parse", ["hub"])])
    snap(2, base + [("src/cli.c", "parse", ["hub"]), ("src/cli.c", "run", ["parse", "util"])])
    snap(3, base[:2] + [("src/cli2.c", "parse", ["hub"]), ("src/cli2.c", "run", ["parse", "util"])])
    return ds


def test_ingest_identity_and_calls():
    ds = make_dataset()
    names = [f.name for f in ds.funcs]
    assert names.count("parse") == 2                      # a move is a death plus a birth, never a silent match
    hub = next(i for i, f in enumerate(ds.funcs) if f.name == "hub")
    assert hub in ds.alive[0] and hub in ds.alive[3]
    callers = {a for a, b in ds.calls[0] if b == hub}
    assert {ds.funcs[a].name for a in callers} == {"util", "main"}
    assert all(a != b for a, b in ds.calls[2])            # no self edges


def test_same_name_in_one_file_gets_distinct_keys():
    ds = collect.Dataset("t", [collect.Snapshot("a", "2020-01-01", "x")])
    cols = ["id", "file_id", "func_name", "complexity", "loc", "start_line", "args", "func_archetype", "calls_out_to"]
    rows = [(1, 1, "f", 1, 5, 1, 0, "", "[]"), (2, 1, "f", 1, 5, 20, 0, "", "[]")]
    collect._ingest(ds, 0, {1: ("a.cpp", "cpp")}, rows, cols)
    assert len(ds.funcs) == 2


def test_runs_and_ages():
    assert scene.runs([0, 1, 1, 0, 1]) == [1, 2, 4, 4]
    assert scene.runs([1, 1]) == [0, 1]
    assert scene.runs([0, 0]) == []
    assert scene.road_ages([0, 2, 3, 0, 1]) == [0, 1, 2, 0, 1]


def test_pack_has_no_overlaps():
    rng = np.random.default_rng(0)
    r = rng.uniform(0.5, 4, 60)
    P, enc = layout.pack(r, gap=0.2)
    d = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
    assert (d + np.eye(60) * 1e9 >= r[:, None] + r[None] - 1e-6).all()
    assert (np.hypot(*P.T) + r <= enc + 1e-6).all()


def test_layout_contains_functions_and_separates_files():
    ds = make_dataset()
    lay = layout.build(ds, iterations=60)
    for i in range(len(ds.funcs)):
        f = lay.func_file[i]
        assert f >= 0
        assert np.hypot(*(lay.func_xy[i] - lay.file_xy[f])) + lay.func_r[i] <= lay.file_r[f] + 1e-6
    d = np.hypot(*(lay.file_xy[:, None] - lay.file_xy[None]).transpose(2, 0, 1))
    n = len(lay.files)
    assert (d + np.eye(n) * 1e9 >= lay.file_r[:, None] + lay.file_r[None] - 0.5).all()


def test_scene_and_html_roundtrip(tmp_path):
    ds = make_dataset()
    lay = layout.build(ds, iterations=60)
    sc = scene.build_scene(ds, lay)
    assert sc.n == 4 and len(sc.captions) == 4
    assert any("moves" in c or "removals" in c or "born" in c for c in sc.captions[1:])
    assert all(len(r.counts) == 4 for r in sc.roads)
    out = html.write_html(sc, tmp_path / "x.html")
    text = out.read_text()
    m = re.search(r"const D=(\{.*?\});\n", text, re.S)
    D = json.loads(m.group(1))
    assert len(D["meta"]["snaps"]) == 4
    assert len(D["funcs"]) == len(ds.funcs)
    assert all(len(f) == 15 for f in D["funcs"])
    assert "__TITLE__" not in text and "/*__DATA__*/" not in text


def test_html_escapes_closing_script_tag(tmp_path):
    ds = make_dataset()
    ds.title = "</script><b>x"
    sc = scene.build_scene(ds, layout.build(ds, iterations=20))
    text = html.write_html(sc, tmp_path / "x.html").read_text()
    assert text.count("</script>") == 1            # only the real closing tag; the title in the data is escaped
    assert "<b>x" not in text.split("<title>")[1].split("</title>")[0]


def git(repo, *a, env=None):
    subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, env=env)


def test_pick_snapshots_even_in_time(tmp_path):
    import os
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "a@b.c")
    git(repo, "config", "user.name", "t")
    # 9 commits clustered in the first month, then one far later: time-spacing must not follow commit count
    dates = [f"2020-01-0{d}T12:00:00" for d in range(1, 9)] + ["2021-01-01T12:00:00"]
    for i, d in enumerate(dates):
        (repo / "f.txt").write_text(str(i))
        git(repo, "add", "-A")
        env = {**os.environ, "GIT_AUTHOR_DATE": d, "GIT_COMMITTER_DATE": d}
        git(repo, "commit", "-q", "-m", f"c{i}", env=env)
    picks = snapshots.pick_snapshots(repo, 3)
    assert 2 <= len(picks) <= 3
    assert picks[0].date.startswith("2020-01") and picks[-1].date.startswith("2021-01")
    dest = tmp_path / "x"
    snapshots.extract(repo, picks[0].commit, dest)
    assert (dest / "f.txt").exists()
    assert not (repo / ".git" / "worktrees").exists()     # user's repo untouched


def test_cli_serve_missing_dir_returns_error(tmp_path):
    from historical_call_graph_gif import cli
    assert cli.main(["serve", str(tmp_path / "nope")]) == 2
