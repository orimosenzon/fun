#!/usr/bin/env bash
# Launcher for the Haskala Flask server. Always runs through the project
# venv so we never accidentally use system Python and hit missing modules
# (python-docx, anthropic, etc.) only after a feature gets clicked.

set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x venv/bin/python ]]; then
    echo "❌ venv לא נמצא ב-$(pwd)/venv" >&2
    echo "   להקים: python3 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

# Verify every requirement is importable. Catches the case where requirements.txt
# was updated but the venv wasn't re-synced — fail fast with a useful message
# instead of crashing mid-request.
missing=$(venv/bin/python - <<'PY'
import importlib, sys
# pip name → import name (only list ones that differ)
aliases = {"python-docx": "docx", "python-dotenv": "dotenv", "pillow": "PIL",
           "google-genai": "google.genai", "pymupdf": "fitz"}
missing = []
with open("requirements.txt") as f:
    for line in f:
        pkg = line.strip().split(">=")[0].split("==")[0].split("[")[0]
        if not pkg or pkg.startswith("#"):
            continue
        mod = aliases.get(pkg, pkg.replace("-", "_"))
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
print("\n".join(missing))
PY
)

if [[ -n "$missing" ]]; then
    echo "❌ חסרות חבילות ב-venv:" >&2
    echo "$missing" | sed 's/^/   - /' >&2
    echo "   להתקין: venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

exec venv/bin/python app.py "$@"
