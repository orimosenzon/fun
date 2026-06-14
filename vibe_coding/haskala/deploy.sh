#!/usr/bin/env bash
# Push the current haskala/ working tree to the HF Space at
# git@hf.co:spaces/orimosenzon/haskala-ocr.
#
# Workflow:
#   1. Pull the Space repo to its local clone (rebases on any browser-edits
#      made on HF, so we never lose remote changes).
#   2. Rsync the deployable files (no venv, no desktop, no logs).
#   3. Commit (msg from CLI arg, otherwise the latest haskala/ commit subject).
#   4. Push over SSH.
#
# Usage:
#   ./deploy.sh                       # auto commit msg
#   ./deploy.sh "highlight feature"   # custom commit msg

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
SPACE="${HASKALA_SPACE_DIR:-$HOME/fun/haskala-space}"

if [[ ! -d "$SPACE/.git" ]]; then
    echo "❌ HF Space clone not found at $SPACE" >&2
    echo "   to clone: git clone git@hf.co:spaces/orimosenzon/haskala-ocr \"$SPACE\"" >&2
    exit 1
fi

REMOTE=$(git -C "$SPACE" remote get-url origin)
if [[ "$REMOTE" != git@hf.co:* ]]; then
    echo "❌ remote of $SPACE is $REMOTE — expected SSH (git@hf.co:...)" >&2
    echo "   to fix: git -C \"$SPACE\" remote set-url origin git@hf.co:spaces/orimosenzon/haskala-ocr" >&2
    exit 1
fi

# Files that actually need to be on the Space. desktop.py, run.sh, deploy.sh,
# venv/, memory/, *.log, __pycache__/, .env stay local.
# README.md on the Space is HF-specific (has the title/sdk/port frontmatter)
# so it lives separately under deploy/space-readme.md and is copied with rename.
FILES=(app.py segmentation.py Dockerfile .dockerignore requirements.txt)
DIRS=(templates rubrics static)

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

# HF Space README has its own frontmatter (title/sdk/port). Tracked
# separately so it doesn't collide with the GitHub-side project README.
if [[ -f "$SRC/deploy/space-readme.md" ]]; then
    cp "$SRC/deploy/space-readme.md" "$SPACE/README.md"
fi

cd "$SPACE"
# --porcelain catches untracked files too; `git diff` alone misses a newly
# added module and would silently skip deploying it.
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

echo "✅ deployed. Space will rebuild automatically: https://huggingface.co/spaces/orimosenzon/haskala-ocr"
