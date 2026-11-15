# repo-roads

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
pip install repo-roads gitgalaxy      # gitgalaxy provides the scanner, 'galaxyscope'
```

`ffmpeg` is optional (needed for MP4 and for the best-quality GIF; without it you still get a GIF and the HTML page).

## Use

```bash
repo-roads make path/to/your/repo -n 24 -o roads-out     # or a clone URL
repo-roads serve roads-out                               # http://127.0.0.1:8765/
```

`make` picks 24 snapshots evenly spaced in time across the first-parent history, extracts each with `git archive`
(your working tree, branches and index are never touched), scans each with GalaxyScope, and writes
`roads-out/<name>.gif`, `.mp4` and `.html`.

Useful options: `--ref main`, `--since 2022-01-01`, `--formats html` (skip the GIF), `--jobs 4`, `--history`
(let GitGalaxy read git history too; slower, adds churn and stability to the file risk scores),
`--keep-scans` (keep the per-snapshot databases), `--max-files 1500`.

Already have a multi-commit GitGalaxy database (one `repo_data` row per commit)?

```bash
repo-roads from-db path/to/master.db --repo curl -n 24
```

## The interactive page

* **Play / pause** (space), step with the arrow keys, drag the slider, change speed.
* **Click a function**: name, file, archetype, lines, complexity, arguments, z-score, public/documented, who calls
  it and what it calls (click to jump), callers over time, its keyword signals, and its file's GitGalaxy risk
  scores. Its callers are drawn in white and its callees in blue.
* **Click a file** for its largest functions and risk scores. **Search** a function by name (Enter).
* Wheel to zoom, drag to pan, double-click to reset. Link straight to a view with `page.html#t=12&f=345`.
* Self-hosting: `repo-roads serve` is a plain static server (`--host 0.0.0.0` to share on your network). The HTML
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

Built and tested up to roughly 1,500 files and 13,000 functions per history (the curl repository); larger
histories keep the busiest files (`--max-files`). Scan time is GalaxyScope's, per snapshot.

## Licence

Not yet chosen. GalaxyScope itself is licensed separately (PolyForm Noncommercial for the community tier); scanning
with it is subject to those terms.
