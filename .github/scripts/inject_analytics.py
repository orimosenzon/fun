#!/usr/bin/env python3
"""Inject the Google Analytics tag into every HTML page at deploy time.

Runs inside the Pages workflow on the checked-out tree, so source files stay
untouched and every new app gets analytics automatically.
Pages that already load gtag are skipped.

  python3 inject_analytics.py [root]          # inject in place
  python3 inject_analytics.py [root] --check  # only report what would change
"""
import re
import sys
from pathlib import Path

GA_ID = "G-GG46M915KH"
SNIPPET = f"""<!-- GA:auto -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_ID}');
</script>
"""
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "site-packages"}
HEAD_OPEN = re.compile(r"<head(\s[^>]*)?>", re.IGNORECASE)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    root = Path(args[0] if args else ".")
    injected, skipped, no_head = [], [], []
    for path in sorted(root.rglob("*.html")):
        if SKIP_DIRS & set(path.parts):
            continue
        try:
            html = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "googletagmanager.com/gtag" in html:
            skipped.append(path)
            continue
        m = HEAD_OPEN.search(html)
        if not m:
            no_head.append(path)
            continue
        injected.append(path)
        if not check:
            path.write_text(html[:m.end()] + "\n" + SNIPPET + html[m.end():], encoding="utf-8")
    verb = "would inject" if check else "injected"
    print(f"{verb}: {len(injected)}, already tagged: {len(skipped)}, no <head>: {len(no_head)}")
    if check:
        for p in no_head:
            print("  no <head>:", p)


if __name__ == "__main__":
    main()
