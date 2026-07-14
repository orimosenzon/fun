#!/usr/bin/env bash
# Deploy scan2 to Cloud Run, always from a committed-and-pushed state.
#
# This is the fix for scan2 having lived only in Cloud Shell with no git
# backup: run this instead of a bare `gcloud run deploy`, and it becomes
# structurally impossible to deploy code that isn't also in GitHub.
#
# Workflow:
#   1. Commit any local changes under this directory (msg from CLI arg,
#      otherwise a default message).
#   2. Push to GitHub (git@github.com:orimosenzon/fun.git).
#   3. gcloud run deploy scan2 --source .
#
# Usage (run from Cloud Shell, inside the `fun` clone):
#   ./deploy.sh                          # auto commit msg
#   ./deploy.sh "fix detect_rotation model id"
#
# One-time setup in Cloud Shell:
#   gh auth login
#   gh repo clone orimosenzon/fun
#   cd fun/haskala/products/scan2
#   # copy in checker.py, main.py, core.py, report.py, requirements.txt

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
cd "$SRC"

PROJECT="${SCAN2_PROJECT:-master-gecko-500709-t0}"
REGION="${SCAN2_REGION:-europe-west1}"
SERVICE="scan2"

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "❌ not inside a git repo — expected to run from within the 'fun' clone" >&2
    exit 1
fi

if [[ -n "$(git status --porcelain .)" ]]; then
    MSG="${1:-update scan2}"
    echo "📝 committing local changes: $MSG"
    git add .
    git commit -m "$MSG"
else
    echo "✅ working tree clean — nothing to commit"
fi

echo "📤 pushing to GitHub…"
git push

echo "🚀 deploying to Cloud Run ($SERVICE, project=$PROJECT, region=$REGION)…"
gcloud run deploy "$SERVICE" \
    --source . \
    --project "$PROJECT" \
    --region "$REGION"

echo "✅ deployed."
