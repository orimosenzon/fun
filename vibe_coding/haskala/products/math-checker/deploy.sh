#!/usr/bin/env bash
# Push the current math-checker/ working tree to the HF Space at
# git@hf.co:spaces/orimosenzon/haskala-math.
#
# Workflow (mirrors products/checker/deploy.sh):
#   1. Pull the Space repo to its local clone (rebases on any browser-edits
#      made on HF, so we never lose remote changes).
#   2. Copy the deployable files (no venv, no memory, no logs).
#   3. Commit (msg from CLI arg, otherwise the latest math-checker/ commit subject).
#   4. Push over SSH.
#
# Usage:
#   ./deploy.sh                       # auto commit msg
#   ./deploy.sh "rotation buttons"    # custom commit msg

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
SPACE="${MATH_CHECKER_SPACE_DIR:-$HOME/fun/haskala-math-space}"

if [[ ! -d "$SPACE/.git" ]]; then
    echo "❌ HF Space clone not found at $SPACE" >&2
    echo "   to clone: git clone git@hf.co:spaces/orimosenzon/haskala-math \"$SPACE\"" >&2
    echo "   (the Space must first be created on HF: huggingface.co/new-space, SDK=Docker)" >&2
    exit 1
fi

REMOTE=$(git -C "$SPACE" remote get-url origin)
if [[ "$REMOTE" != git@hf.co:* ]]; then
    echo "❌ remote of $SPACE is $REMOTE — expected SSH (git@hf.co:...)" >&2
    echo "   to fix: git -C \"$SPACE\" remote set-url origin git@hf.co:spaces/orimosenzon/haskala-math" >&2
    exit 1
fi

# Files that actually need to be on the Space. run.sh, deploy.sh, venv/,
# memory/, *.log, __pycache__/, .env stay local. README.md on the Space is
# HF-specific (title/sdk/port frontmatter), kept under deploy/space-readme.md.
#
# core.py and report.py were part of app.py until the 2026-07-03 refactor and
# were missing from this list until 2026-08-05 — deploying in between would
# have pushed an app.py whose imports were not on the Space.
FILES=(app.py core.py report.py Dockerfile .dockerignore requirements.txt)
DIRS=(templates static haskala_grading)

# Vendor the shared grading library into this directory before deploying.
# core.py/report.py here are thin shims over haskala_grading.math_core /
# .math_report, which live in haskala/lib/ and are shared with
# math-form-checker. Only this directory is synced to the Space, so a sibling
# lib/ would simply be absent and the container would die on import.
#
# The copy is gitignored (haskala/.gitignore: products/*/haskala_grading/).
# Never edit the vendored copy — it is overwritten here on every deploy.
echo "📦 vendoring shared library haskala_grading/…"
rsync -a --delete --exclude='__pycache__' \
      "$SRC/../../lib/haskala_grading/" "$SRC/haskala_grading/"

echo "📥 pulling latest from HF Space ($SPACE)…"
git -C "$SPACE" pull --rebase --autostash

echo "📤 syncing files from $SRC → $SPACE"
for f in "${FILES[@]}"; do
    if [[ -f "$SRC/$f" ]]; then
        cp "$SRC/$f" "$SPACE/$f"
    else
        echo "⚠️  $SRC/$f missing — skipping"
    fi
done
for d in "${DIRS[@]}"; do
    if [[ -d "$SRC/$d" ]]; then
        rsync -a --delete \
              --exclude='__pycache__' \
              --exclude='*.pyc' \
              "$SRC/$d/" "$SPACE/$d/"
    else
        echo "⚠️  $SRC/$d missing — skipping"
    fi
done

# HF Space README has its own frontmatter (title/sdk/port).
if [[ -f "$SRC/deploy/space-readme.md" ]]; then
    cp "$SRC/deploy/space-readme.md" "$SPACE/README.md"
fi

cd "$SPACE"
# --porcelain catches untracked files too; `git diff` alone would miss a
# newly added module and silently skip deploying it.
if [[ -z "$(git status --porcelain)" ]]; then
    echo "✅ nothing changed — Space already in sync"
    exit 0
fi

if [[ $# -ge 1 && -n "$1" ]]; then
    MSG="$1"
else
    MSG=$(git -C "$SRC" log -1 --pretty=%s -- . 2>/dev/null || echo "deploy update")
fi

echo "📝 commit: $MSG"
git add -A
git commit -m "$MSG"

echo "🚀 pushing to HF…"
git push

echo "✅ deployed. Space will rebuild automatically: https://huggingface.co/spaces/orimosenzon/haskala-math"
