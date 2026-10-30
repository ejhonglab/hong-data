"""Drives the app in headless Chrome (Playwright), by commands from stdin, one a line.

    .claude/skills/run-hong-data/venv/bin/python .claude/skills/run-hong-data/driver.py \
        --base http://localhost:8765/ --out <folder for screenshots> <<'EOF'
    nav #headspace
    ss start
    EOF

Arguments are split as by a shell (shlex): quote a selector with spaces in it. Selectors
are Playwright's (CSS; `text=...`; `>> nth=-1` for the last match); a command acts on
the first match.

    nav <url, or #hash of --base>
    wait <selector>             until it is visible (at most 2 min: a first calculation looks things up)
    wait-fn <JS expression>     until it is true (the rest of the line, as is)
    click | hover | focus <selector>
    fill <selector> <text>      types into an input, as typing does (input events)
    select <selector> <value>   chooses an option of a <select>
    press <key>                 e.g. Enter, Escape, ArrowDown, Tab
    move <x> <y>                the mouse, e.g. away from what it hovers
    text <selector>             prints its text
    visible <selector>          prints whether it is visible
    attr <selector> <name>      prints an attribute
    eval <JS expression>        prints its value, as JSON (the rest of the line, as is)
    viewport <width> <height>
    ss <name>                   screenshot of the window to <out>/<name>.png (each tab scrolls itself, not
                                the page: a taller viewport shows more of it)
    errors                      prints console errors and uncaught exceptions, so far

Lines starting with # are comments. The first command that fails stops it (exit 1),
with a screenshot of the window then (<out>/error.png). Console errors are printed at
the end in any case.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

WAIT_MS = 120_000


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="http://localhost:8765/")
    p.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "hong-data-driver"))
    p.add_argument("--width", type=int, default=1400)
    p.add_argument("--height", type=int, default=900)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # (Chrome as installed, else Playwright's own: `playwright install chromium`)
    chrome = os.environ.get("CHROME") or next((c for c in ("/usr/bin/google-chrome", "/usr/bin/chromium") if os.path.exists(c)), None)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chrome, headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(f"uncaught: {e}"))
        page.on("console", lambda m: errors.append(f"console: {m.text}") if m.type == "error" else None)
        status = 0
        for n, line in enumerate(sys.stdin, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cmd, _, rest = line.partition(" ")
            try:
                a = shlex.split(rest) if cmd not in ("eval", "wait-fn") else [rest]
                if cmd == "nav":
                    page.goto(args.base.rstrip("/") + "/" + a[0] if a[0].startswith("#") else a[0])
                elif cmd == "wait":
                    page.wait_for_selector(a[0], state="visible", timeout=WAIT_MS)
                elif cmd == "wait-fn":
                    page.wait_for_function(a[0], timeout=WAIT_MS)
                elif cmd in ("click", "hover", "focus"):
                    getattr(page, cmd)(a[0])
                elif cmd == "fill":
                    page.fill(a[0], a[1])
                elif cmd == "select":
                    page.select_option(a[0], a[1])
                elif cmd == "press":
                    page.keyboard.press(a[0])
                elif cmd == "move":
                    page.mouse.move(float(a[0]), float(a[1]))
                elif cmd == "text":
                    print(page.inner_text(a[0]))
                elif cmd == "visible":
                    print(page.is_visible(a[0]))
                elif cmd == "attr":
                    print(page.get_attribute(a[0], a[1]))
                elif cmd == "eval":
                    print(json.dumps(page.evaluate(a[0]), ensure_ascii=False))
                elif cmd == "viewport":
                    page.set_viewport_size({"width": int(a[0]), "height": int(a[1])})
                elif cmd == "ss":
                    path = out / f"{a[0]}.png"
                    page.screenshot(path=str(path))
                    print(f"screenshot: {path}")
                elif cmd == "errors":
                    print("\n".join(errors) or "no errors")
                else:
                    raise ValueError(f"unknown command {cmd!r}")
            except Exception as e:
                print(f"line {n}: {line}\n  failed: {str(e).splitlines()[0]}", file=sys.stderr)
                page.screenshot(path=str(out / "error.png"))
                print(f"screenshot: {out / 'error.png'}", file=sys.stderr)
                status = 1
                break
        if errors:
            print("console errors:\n  " + "\n  ".join(errors), file=sys.stderr)
        browser.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
