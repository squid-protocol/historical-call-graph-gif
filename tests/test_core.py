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


def _commit(repo, d, msg, allow_empty=False):
    import os
    env = {**os.environ, "GIT_AUTHOR_DATE": d, "GIT_COMMITTER_DATE": d}
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg, *(["--allow-empty"] if allow_empty else []), env=env)


def _repo_with_empty_first_commit(tmp_path):
    repo = tmp_path / "svn"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "a@b.c")
    git(repo, "config", "user.name", "t")
    _commit(repo, "2003-01-01T12:00:00", "empty import", allow_empty=True)     # same timestamp as its child
    for i, d in enumerate(["2003-01-01T12:00:00", "2010-01-01T12:00:00", "2020-01-01T12:00:00"]):
        (repo / "f.txt").write_text(str(i))
        _commit(repo, d, f"c{i}")
    return repo


def test_empty_leading_commits_are_skipped(tmp_path):
    repo = _repo_with_empty_first_commit(tmp_path)
    picks = snapshots.pick_snapshots(repo, 4)
    assert picks and all(not snapshots.tree_is_empty(repo, p.commit) for p in picks)
    assert picks[0].date.startswith("2003")


def test_extract_of_empty_commit_is_a_clean_error(tmp_path):
    repo = _repo_with_empty_first_commit(tmp_path)
    first = snapshots.list_commits(repo)[0][1]
    with pytest.raises(RuntimeError, match="no files"):
        snapshots.extract(repo, first, tmp_path / "x")


def test_scan_all_skips_a_failed_snapshot(tmp_path, monkeypatch):
    from historical_call_graph_gif import scan
    monkeypatch.setattr(scan, "find_scanner", lambda: "fake")

    def fake(repo, pick, idx, work, scanner, history):
        if idx == 1:
            raise RuntimeError("boom\ndetail")
        db = work / f"{idx}.db"
        db.write_text("x")
        return db
    monkeypatch.setattr(scan, "scan_snapshot", fake)
    picks = [snapshots.Pick(f"c{i}", f"2020-0{i + 1}-01", f"s{i}") for i in range(3)]
    msgs = []
    ok_picks, dbs = scan.scan_all(tmp_path, picks, tmp_path / "w", jobs=1, progress=msgs.append)
    assert [p.label for p in ok_picks] == ["s0", "s2"] and len(dbs) == 2
    assert any("skipped 2/3" in m for m in msgs)
    monkeypatch.setattr(scan, "scan_snapshot", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError, match="no snapshot"):
        scan.scan_all(tmp_path, picks, tmp_path / "w2", jobs=1, progress=lambda m: None)


def test_extract_skips_absolute_symlinks_and_keeps_files(tmp_path):
    import os
    repo = tmp_path / "lnk"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "a@b.c")
    git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("def f():\n    pass\n")
    os.symlink("/etc/passwd", repo / "abs_link")          # absolute target: a strict extractor aborts here
    os.symlink("../../outside", repo / "esc_link")
    _commit(repo, "2020-01-01T12:00:00", "with links")
    sha = snapshots.list_commits(repo)[-1][1]
    dest = tmp_path / "out"
    snapshots.extract(repo, sha, dest)
    assert (dest / "a.py").exists()
    assert not (dest / "abs_link").exists() and not (dest / "esc_link").exists()


def test_drop_leading_empty_snapshots():
    ds = make_dataset(4)
    ds.alive[0] = set()                                    # pretend the first snapshot had no functions yet
    n_before = len(ds.snaps)
    collect.drop_leading_empty(ds)
    assert len(ds.snaps) == n_before - 1 and len(ds.alive) == len(ds.calls) == len(ds.snaps)
    assert all(f.first >= 0 for f in ds.funcs)


def test_fast_pack_has_no_overlaps_and_is_fast():
    import time
    rng = np.random.default_rng(1)
    r = rng.uniform(0.5, 3.5, 1500)
    t0 = time.time()
    P, enc = layout.pack(r, gap=0.25)
    assert time.time() - t0 < 20
    R = r + 0.25
    d = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
    assert (d + np.eye(len(r)) * 1e9 >= R[:, None] + R[None] - 1e-6).all()
    assert (np.hypot(*P.T) + R <= enc + 1e-6).all()
    area_ratio = (np.pi * (R ** 2).sum()) / (np.pi * enc ** 2)
    assert area_ratio > 0.35                             # still reasonably dense, not a sparse scatter


def test_layout_scales_to_hundreds_of_files_without_overlap():
    import time
    rng = np.random.default_rng(2)
    snaps = [collect.Snapshot("a", "2020-01-01", "x"), collect.Snapshot("b", "2021-01-01", "y")]
    ds = collect.Dataset("big", snaps)
    cols = ["id", "file_id", "func_name", "complexity", "loc", "start_line", "args", "func_archetype", "calls_out_to"]
    rows, fm = [], {}
    for f in range(400):
        path = f"d{f % 12}/sub{f % 5}/f{f}.py"
        fm[f + 1] = (path, "python")
        for k in range(int(rng.integers(1, 30))):
            callees = [f"fn_{int(rng.integers(0, 400))}_0"] if k else []
            rows.append((len(rows), f + 1, f"fn_{f}_{k}", 2, 10 + k, k, 1, "", json.dumps(callees)))
    for s in range(2):
        collect._ingest(ds, s, fm, rows, cols)
    t0 = time.time()
    lay = layout.build(ds)
    assert time.time() - t0 < 120
    n = len(lay.files)
    d = np.hypot(*(lay.file_xy[:, None] - lay.file_xy[None]).transpose(2, 0, 1))
    assert (d + np.eye(n) * 1e9 >= lay.file_r[:, None] + lay.file_r[None] - 1.5).all()


def test_road_age_is_measured_in_years_not_snapshots():
    years = [2000.0, 2010.0, 2020.0]
    assert scene.road_age_years([1, 1, 1], years) == [0.0, 10.0, 20.0]
    assert scene.road_age_years([1, 0, 1], years) == [0.0, 0.0, 0.0]     # a gap resets the age
    assert scene.age_full_years([2000.0, 2026.0]) == 10.0
    assert scene.age_full_years([2020.0, 2023.0]) == pytest.approx(1.2)
