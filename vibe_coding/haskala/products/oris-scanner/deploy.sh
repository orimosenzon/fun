#!/usr/bin/env bash
# Deploy oris-scanner to Cloud Run, always from a committed-and-pushed state.
# Mirrors products/scan2/deploy.sh (same reasoning: no code-only-in-Cloud-Shell).
#
# Workflow:
#   1. Vendor haskala/lib/haskala_grading/ into this directory (see below).
#   2. Commit any local changes under this directory (msg from CLI arg,
#      otherwise a default message).
#   3. Push to GitHub (git@github.com:orimosenzon/fun.git).
#   4. gcloud run deploy oris-scanner --source .
#
# Vendoring runs first, and deliberately so: a change to the shared library is
# then already in place when the commit step runs, so what gets deployed and
# what is in git describe the same code.
#
# NOTE: this only (re)deploys the Cloud Run service/code. It does NOT touch
# the Cloud Scheduler trigger — see setup_infra.sh for the one-time trigger
# wiring (scheduler job, IAM).
#
# Usage (run from Cloud Shell, inside the `fun` clone):
#   ./deploy.sh                          # auto commit msg
#   ./deploy.sh "tune language rubric prompt"
#
# Memory is pinned here, not left to the service's existing setting. 512Mi —
# the Cloud Run default — is not enough: on 2026-07-28 a single-page submission
# (a 2.1MB PDF whose page renders large at 200 DPI) pushed the container past
# 900MiB and got it OOM-killed mid-check, ~40 times overnight, because the
# dedup ledger kept reclaiming the abandoned claim. 2Gi cleared it on the first
# try. Keeping it in this script means a rebuild from scratch can't silently
# regress to 512Mi.
#
# Log verbosity: INFO by default (the trace of submissions actually graded).
# Deploy with ORIS_SCANNER_LOG_LEVEL=DEBUG to also get the every-5-minutes
# polling chatter. Note --set-env-vars replaces the whole env block, so a
# LOG_LEVEL set by hand via `gcloud run services update` is reset on the next
# deploy — that's why it's pinned here rather than left out. Read the logs
# with ./logs.sh.

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
cd "$SRC"

# Vendor the shared grading library into this directory before deploying.
#
# core.py and report.py live in haskala/lib/haskala_grading/ and are shared with
# form-checker. `gcloud run deploy --source .` uploads only this directory, so a
# sibling lib/ is simply absent from the build — the container would start and
# then fail on `from haskala_grading import core`. Copying it in here is what
# makes the shared library reach the container at all.
#
# The copy is gitignored (see haskala/.gitignore): git's copy of the library is
# lib/, and only lib/. --delete matters — without it a module deleted upstream
# lingers here forever and the container keeps running code that no longer
# exists in git.
#
# Note --source respects .gcloudignore, not .gitignore, so being gitignored does
# not stop it being uploaded. If a .gcloudignore is ever added to this
# directory, it must not exclude haskala_grading/.
echo "📦 vendoring shared library haskala_grading/…"
rsync -a --delete --exclude='__pycache__' "$SRC/../../lib/haskala_grading/" "$SRC/haskala_grading/"

PROJECT="${ORIS_SCANNER_PROJECT:-master-gecko-500709-t0}"
REGION="${ORIS_SCANNER_REGION:-europe-west1}"
SERVICE="oris-scanner"

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "❌ not inside a git repo — expected to run from within the 'fun' clone" >&2
    exit 1
fi

if [[ -n "$(git status --porcelain .)" ]]; then
    MSG="${1:-update oris-scanner}"
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
    --function=process_my_drive_files \
    --project "$PROJECT" \
    --region "$REGION" \
    --no-allow-unauthenticated \
    --memory=2Gi \
    --set-secrets="GEMINI_API_KEY=gemini-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,GROQ_API_KEY=groq-api-key:latest,AZURE_OPENAI_API_KEY=azure-openai-api-key:latest" \
    --set-env-vars="AZURE_OPENAI_ENDPOINT=https://haskala-foundry-resource.openai.azure.com/openai/v1,AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini,LOG_LEVEL=${ORIS_SCANNER_LOG_LEVEL:-INFO}"

echo "✅ deployed. Run setup_infra.sh once (first deploy only) to wire the Cloud Scheduler trigger."
