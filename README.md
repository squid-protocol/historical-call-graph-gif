# historical-call-graph-gif

Turn a git repository's history into a road map of its call network:

* an animated **GIF** and **MP4**, and
* a **self-hosted interactive page** (one HTML file, no network needed) you can pause, scrub, zoom and click.

Every file is a circle, every function a dot in it. Files that call each other sit near each other, and calls between
files are merged into **roads**: road width is how many calls it carries, road colour is its age (green when new,
blue when old). New functions flash green, removed ones flash amber. Click any function to see what
[GitGalaxy](https://gitgalaxy.io) knows about it.

The layout is computed once for the whole history, so nothing moves between frames: you see the same city
being built, widened and rebuilt.

## Install

```bash
pip install historical-call-graph-gif gitgalaxy      # gitgalaxy provides the scanner, 'galaxyscope'
```

The short command `hcg-gif` is an alias for `historical-call-graph-gif`. `ffmpeg` is optional (needed for MP4 and for the best-quality GIF; without it you still get a GIF and the HTML page).

## Use

```bash
historical-call-graph-gif make path/to/your/repo -o roads-out     # or a clone URL
historical-call-graph-gif serve roads-out                               # http://127.0.0.1:8765/
```

`make` chooses how many snapshots to take from the history itself (see below), extracts each with `git archive`
(your working tree, branches and index are never touched), scans each with GalaxyScope, and writes
`roads-out/<name>.gif`, `.mp4` and `.html`.

### How many snapshots, and when

By default the count is **automatic**: about one snapshot per quarter for the first six years of history and
roughly one per 2.5 quarters after that (a 3-year repo gets 12, a 26-year one up to 56), never more than the number
of quarters that actually had commits, and capped by repository size because every snapshot is a full scan (48 for
small repositories, 32 for mid-sized, 20 for big). The choice is printed so you can see why.

```bash
historical-call-graph-gif make repo -n 40            # a fixed number
historical-call-graph-gif make repo --every 6m       # one per interval: 90d, 2w, 3m, 1y
historical-call-graph-gif make repo --spacing activity   # evenly in commit count: busy periods get more frames
historical-call-graph-gif make repo --since 2022-01-01   # only part of the history
```

Snapshots are spaced evenly in time by default, so a quiet year and a busy year get the same room. Labels
follow the spacing (quarters, months or dates).

Other options: `--ref main`, `--until`, `--formats html` (skip the GIF), `--jobs 4`, `--history`
(let GitGalaxy read git history too; slower, adds churn and stability to the file risk scores),
`--keep-scans` (keep the per-snapshot databases), `--max-files 4000`.

Already have a multi-commit GitGalaxy database (one `repo_data` row per commit)?

```bash
historical-call-graph-gif from-db path/to/master.db --repo curl -n 24
```

## The interactive page

* **Play / pause** (space), step with the arrow keys, drag the slider, change speed.
* **Click a function**: name, file, archetype, lines, complexity, arguments, z-score, public/documented, who calls
  it and what it calls (click to jump), callers over time, its keyword signals, and its file's GitGalaxy risk
  scores. Its callers are drawn in white and its callees in blue.
* **Click a file** for its largest functions and risk scores. **Search** a function by name (Enter).
* Wheel to zoom, drag to pan, double-click to reset. Link straight to a view with `page.html#t=12&f=345`.
* Self-hosting: `historical-call-graph-gif serve` is a plain static server (`--host 0.0.0.0` to share on your network). The HTML
  file also works opened directly from disk or dropped on any static host.

Metrics in the panel are from the last snapshot the function appeared in.

## How to read it

* A function is identified by (file, name). **A function that moves file shows as a death plus a birth.** Captions
  report what share of the quarter's dead reappear under the same name, so a reorganisation reads as a move, not
  as churn.
* A *road* is all calls between one pair of files, merged. Calls are matched by name (same file first, otherwise a
  name that is unique in the snapshot), so ambiguous names are skipped.
* Positions are a force-directed sketch: heavily connected files are pulled towards the centre, but distance is a
  hint, not a measurement.
* Roads follow a small street network (junctions between nearby files, express lines for the busiest pairs) so
  that neighbours get direct routes and only long trips share highways.

## Limits

Tested on C, C++, Java, Python, Go, Rust, Zig, TypeScript, Ruby, PHP, Perl, MATLAB, Fortran and Scheme
repositories, up to about 100,000 functions (OpenCV). Histories with more than `--max-files` files keep the files
with the most functions and say how many are not drawn. Scan time is GalaxyScope's, per snapshot.

## Licence

This tool is licensed under the [Apache License 2.0](LICENSE).

It runs the GitGalaxy scanner (`galaxyscope`) as a separate program and does not include or link to any GitGalaxy
code. **The scanner is licensed separately** (PolyForm Noncommercial 1.0.0 for the community tier), and scanning
with it is subject to those terms: using this tool commercially means holding a commercial GitGalaxy licence key.
The optional `ffmpeg` is likewise a separate program under its own licence.
