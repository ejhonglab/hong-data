---
name: run-hong-data
description: Run, start, drive and screenshot the hong-data odor response database web app (Upload, Browse, Compare, Odors, Headspace calculator tabs) in headless Chrome, call its HTTP API with curl, or run its tests. Use when asked to run or start the app or its server, check a UI change in the browser, take a screenshot of a tab, click through it, or run the test suite.
---

The app is a Starlette server (`app/main.py`) serving one page (`app/static/`) whose tabs
are hash routes (`#browse`, `#headspace`, `#odors/<id>`, ...). Agents start it with
`serve.sh` against a **scratch data folder**, then drive it with `driver.py`: Playwright
on the installed Google Chrome, reading commands from stdin. All paths below are
relative to the repo root.

## Prerequisites

- The project's `./venv` (README, "Development": `pip install -r requirements-dev.txt`,
  which needs git for chemutils). Use `./venv/bin/python`: there is no `python` on PATH.
- Google Chrome at `/usr/bin/google-chrome` (or set `CHROME=<path>`). `chromium-cli` is
  not installed here.
- The driver's own venv, once (gitignored):

```bash
python3 -m venv .claude/skills/run-hong-data/venv && .claude/skills/run-hong-data/venv/bin/pip install -q playwright
```

No `playwright install` is needed: it drives the installed Chrome.

## Run (agent path)

Start the server on port 8765, with a scratch data folder. It is made if it does not
exist, and three odors are added to its odor table (`seed.py`):

```bash
OLFDB_DATA_DIR=<scratch>/olfdb-run .claude/skills/run-hong-data/serve.sh start
```

Drive it. This heredoc adds two odors to the Headspace calculator, calculates, hovers
over a result and takes screenshots:

```bash
.claude/skills/run-hong-data/venv/bin/python .claude/skills/run-hong-data/driver.py --out <scratch>/shots <<'EOF'
nav #headspace
wait #hs-search
fill #hs-search "ethyl acetate"
wait "#hs-suggestions [data-id]"
press Enter
fill "#hs-components input[data-conc] >> nth=-1" -3
fill #hs-search 2-heptanone
wait "#hs-suggestions [data-id]"
press Enter
fill "#hs-components input[data-conc] >> nth=-1" -2
click #hs-calc
wait "#hs-results .hs-why"
ss results
hover "[data-derive='0']"
text #hs-derivation
ss popup
errors
EOF
```

Screenshots go to `<out>/<name>.png`; **look at them**. The commands (`nav`, `wait`,
`wait-fn`, `click`, `hover`, `focus`, `fill`, `select`, `press`, `move`, `point`, `text`,
`visible`, `attr`, `eval`, `viewport`, `ss`, `errors`) are listed in `driver.py`'s
docstring. Arguments are split as a shell would split them, so quote selectors that
contain spaces. The first failing command stops the run (exit 1) and saves
`<out>/error.png`. Console errors are printed at the end of every run.

The HTTP API, with the server up (every endpoint is in design.md, "HTTP API"):

```bash
curl -s -X POST http://localhost:8765/api/headspace -H 'Content-Type: application/json' -d '{"solvent": "paraffin oil", "components": [{"odor_id": 1, "log10_conc": -3}]}'
```

Stop it:

```bash
.claude/skills/run-hong-data/serve.sh stop
```

## Tests

```bash
OLFDB_DATA_DIR=<scratch>/olfdb-test ./venv/bin/python -m pytest tests/ -q
```

228 pass in about 30 s. For internals, call a module directly (e.g.
`headspace.calculate`), as `tests/test_headspace.py` does.

## Run (human path)

README, "Development": `uvicorn app.main:app --reload --port 8080` uses `./data`, the
real data. Agents use `serve.sh` instead.

## Gotchas

- **`OLFDB_DATA_DIR` must always be set**: for the server, pytest, and any script that
  imports `app`. Without it, `app/config.py` uses `./data`, the real database. `serve.sh`
  and `seed.py` refuse to run without it or when it points at `data/`.
- The real odor table may be empty, and the Headspace calculator and Odors tab need
  odors, so `seed.py` adds ethyl acetate, 2-heptanone and 1-octen-3-ol as an odor sheet
  would (`odormeta.parse_sheet`, `db.apply_odor_sync`). Edit its `ODORS` to add more. It
  only adds odors that aren't there yet.
- The first calculation for an odor looks up its properties through chemutils (PubChem,
  so it needs the network) and keeps them in the scratch database. Here the whole flow
  above took about 4 s.
- Every tab's section is in the page at once, hidden, so a bare `h1` matches 9 elements
  and `click h1` times out after 30 s ("element is not visible"). Scope selectors to
  `#tab-<name>`.
- Each tab scrolls itself, not the document (`#tab-headspace { overflow: auto }`), so a
  full-page screenshot is just the window. Use `viewport 1400 2600` before `ss` to see a
  whole tab. Playwright scrolls an element into view itself before clicking or hovering.
- `hover` and `click` scroll their target to an edge of the window first, and the default
  window (1400×900) is tall, so they can hide bugs that depend on where something is in
  the window: a popup by a value mid-window in a short window covered its value and hid
  itself, which never showed this way. To test where something is, scroll its tab
  (`eval document.querySelector('#tab-headspace').scrollTop = 400`), then `point
  <selector>` (the mouse to it where it is), in a smaller `viewport` too (e.g. 1366 640).
- Each driver run is a fresh browser, so whatever the page keeps in localStorage (the
  mixture, units, Settings) starts empty every time.
- A script run from outside the repo that imports `app` needs `PYTHONPATH=.`.

## Troubleshooting

- `ModuleNotFoundError: No module named 'app'` when running a script → run it with
  `PYTHONPATH=<repo>`, as `serve.sh` does for `seed.py`.
- `/bin/bash: python: command not found` → use `./venv/bin/python` (or `python3` for the
  driver's venv).
- `Page.click: Timeout 30000ms exceeded` with "element is not visible" in the log → the
  selector matched an element of a hidden tab: scope it to `#tab-<name>`.
- `port 8765 is in use` → `serve.sh stop` (or pass another port: `serve.sh start 8766`,
  `driver.py --base http://localhost:8766/`).
- The server did not come up → `serve.sh` prints the end of its log, `<data dir>.log`
  (the data folder's path plus `.log`).
