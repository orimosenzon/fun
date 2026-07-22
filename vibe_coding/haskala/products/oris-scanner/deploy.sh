#!/usr/bin/env bash
# Deploy oris-scanner to Cloud Run, always from a committed-and-pushed state.
# Mirrors products/scan2/deploy.sh (same reasoning: no code-only-in-Cloud-Shell).
#
# Workflow:
#   1. Commit any local changes under this directory (msg from CLI arg,
#      otherwise a default message).
#   2. Push to GitHub (git@github.com:orimosenzon/fun.git).
#   3. gcloud run deploy oris-scanner --source .
#
# NOTE: this only (re)deploys the Cloud Run service/code. It does NOT touch
# the Cloud Scheduler trigger — see setup_infra.sh for the one-time trigger
# wiring (scheduler job, IAM).
#
# Usage (run from Cloud Shell, inside the `fun` clone):
#   ./deploy.sh                          # auto commit msg
#   ./deploy.sh "tune language rubric prompt"

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
cd "$SRC"

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
    --set-secrets="GEMINI_API_KEY=gemini-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,GROQ_API_KEY=groq-api-key:latest,AZURE_OPENAI_API_KEY=azure-openai-api-key:latest" \
    --set-env-vars="AZURE_OPENAI_ENDPOINT=https://haskala-foundry-resource.openai.azure.com/openai/v1,AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini"

echo "✅ deployed. Run setup_infra.sh once (first deploy only) to wire the Cloud Scheduler trigger."
