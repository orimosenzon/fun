#!/usr/bin/env bash
# מפעיל את שרת math-checker דרך ה-venv המקומי (פורט 5051).
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x venv/bin/python ]]; then
    echo "❌ venv לא נמצא ב-$(pwd)/venv" >&2
    echo "   להקים: python3 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi
exec venv/bin/python app.py "$@"
