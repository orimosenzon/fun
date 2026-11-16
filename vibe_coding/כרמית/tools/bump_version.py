"""Stamps a fresh version on every local script and stylesheet reference in app/ (?v=<stamp>), so a phone that
loads the page after a deploy gets this deploy's files and never a mix of new and cached old ones.
Run before every commit that changes app/:  python3 tools/bump_version.py"""
import re
import time
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
STAMP = time.strftime("%Y%m%d%H%M")
# admin.js, script.js, fb.js and style.css, with or without an earlier ?v=
REF = re.compile(r"""((?:src|href)="|from '\./)((?:admin|script|fb)\.js|style\.css)(?:\?v=\w+)?(["'])""")

for name in ["index.html", "admin.html", "script.js", "admin.js"]:
    path = APP / name
    text = path.read_text(encoding="utf-8")
    new, n = REF.subn(lambda m: f"{m[1]}{m[2]}?v={STAMP}{m[3]}", text)
    if new != text:
        path.write_text(new, encoding="utf-8")
    print(f"{name}: {n} references → v={STAMP}")
