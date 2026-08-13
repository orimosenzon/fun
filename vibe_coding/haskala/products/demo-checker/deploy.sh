#!/usr/bin/env bash
# Deploy demo-checker to Cloud Run, always from a committed-and-pushed state.
# Mirrors products/oris-scanner/deploy.sh (same reasoning: no code that only
# exists on someone's laptop or in Cloud Shell).
#
# Workflow:
#   1. Vendor haskala/lib/haskala_grading/ into this directory (see below).
#   2. Commit any local changes under this directory (msg from CLI arg,
#      otherwise a default message).
#   3. Push to GitHub (git@github.com:orimosenzon/fun.git).
#   4. gcloud run deploy demo-checker --source .
#
# NOTE: this only (re)deploys the Cloud Run service/code. It does NOT touch the
# Cloud Scheduler trigger — see setup_infra.sh for the one-time trigger wiring.
#
# Usage:
#   ./deploy.sh                          # auto commit msg
#   ./deploy.sh "widen the rubric dropdown"
#
# Memory is pinned at 2Gi, not left to the Cloud Run default of 512Mi. The
# lesson is inherited from oris-scanner and applies identically here, because
# it is the same rendering code: on 2026-07-28 a single-page 2.1MB PDF rendered
# at 200 DPI pushed that container past 900MiB and got it OOM-killed ~40 times
# overnight. Pinning it here means a rebuild from scratch cannot regress.

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
cd "$SRC"

# Vendor the shared grading library into this directory before deploying.
#
# core.py and report.py live in haskala/lib/haskala_grading/ and are shared with
# oris-scanner. `gcloud run deploy --source .` uploads only this directory, so a
# sibling lib/ is simply absent from the build — the container would start and
# then fail on `from haskala_grading import core`.
#
# The copy is gitignored (see haskala/.gitignore): git's copy of the library is
# lib/, and only lib/. Never edit the vendored copy — it is overwritten here on
# every deploy. --delete matters: without it a module deleted upstream lingers
# and the container keeps running code that no longer exists in git.
echo "📦 vendoring shared library haskala_grading/…"
rsync -a --delete --exclude='__pycache__' "$SRC/../../lib/haskala_grading/" "$SRC/haskala_grading/"

PROJECT="${DEMO_CHECKER_PROJECT:-master-gecko-500709-t0}"
REGION="${DEMO_CHECKER_REGION:-europe-west1}"
SERVICE="demo-checker"

# Same fallback logs.sh carries: the SDK is installed under $HOME rather than
# system-wide, and its PATH entry only exists in interactive shells that source
# the profile. Without this the script does everything — vendors, commits,
# pushes — and only then dies on `gcloud: command not found`, leaving a commit
# pushed for a deploy that never happened.
if ! command -v gcloud >/dev/null 2>&1 && [[ -x "$HOME/google-cloud-sdk/bin/gcloud" ]]; then
    PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi
if ! command -v gcloud >/dev/null 2>&1; then
    echo "❌ gcloud not found on PATH (looked in \$HOME/google-cloud-sdk/bin too)" >&2
    exit 1
fi

# The Form's linked responses spreadsheet — the id between /d/ and /edit in its
# URL. The service cannot do anything without it, so fail here with an
# explanation rather than deploying something that only fails at the first poll.
# Already deployed once? Then the id the service is running with is the answer,
# and asking for it again is a trap: --set-env-vars replaces the block
# wholesale, so a redeploy that omits it does not keep the old value, it wipes
# it. Recovering it from the live service makes a routine redeploy a one-word
# command and keeps the error below for what it is really about — the first
# deploy, when there is nothing to recover it from.
if [[ -z "${RESPONSES_SHEET_ID:-}" ]]; then
    # The filter yields a one-element list and gcloud prints it as ['…'];
    # no --format spelling coaxes a bare scalar out of it, hence the tr.
    RESPONSES_SHEET_ID="$(gcloud run services describe "$SERVICE" \
        --project "$PROJECT" --region "$REGION" \
        --format='value(spec.template.spec.containers[0].env.filter("name:RESPONSES_SHEET_ID").extract(value))' \
        2>/dev/null | tr -d "[]' " || true)"
    if [[ -n "$RESPONSES_SHEET_ID" ]]; then
        echo "📄 reusing RESPONSES_SHEET_ID from the deployed service: ${RESPONSES_SHEET_ID:0:12}…"
    fi
fi

if [[ -z "${RESPONSES_SHEET_ID:-}" ]]; then
    echo "❌ RESPONSES_SHEET_ID is not set and no deployed service to read it from." >&2
    echo "   Open the Form → Responses → the Sheets icon, and take the id from the" >&2
    echo "   spreadsheet URL (the long token between /d/ and /edit). Then:" >&2
    echo "     RESPONSES_SHEET_ID=1AbC…xyz ./deploy.sh" >&2
    exit 1
fi

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "❌ not inside a git repo — expected to run from within the 'fun' clone" >&2
    exit 1
fi

if [[ -n "$(git status --porcelain .)" ]]; then
    MSG="${1:-update demo-checker}"
    echo "📝 committing local changes: $MSG"
    git add .
    git commit -m "$MSG"
else
    echo "✅ working tree clean — nothing to commit"
fi

echo "📤 pushing to GitHub…"
git push

echo "🚀 deploying to Cloud Run ($SERVICE, project=$PROJECT, region=$REGION)…"
# FEEDBACK_FORM_URL is deliberately NOT in the list below. Writing VAR=${VAR:-}
# sets the key to an EMPTY STRING rather than leaving it unset, and
# os.environ.get's default only fires on an absent key — so listing it silently
# overrode mailer.py's default and mailed three teachers a "Your Feedback:"
# heading with nothing under it. Anything whose fallback lives in the code
# belongs out of this list, or the code's default is dead on Cloud Run.
#
# --set-env-vars replaces the whole env block, so everything the service reads
# has to be listed here — a value set by hand with `gcloud run services update`
# is wiped on the next deploy. That is why the guardrails are pinned in this
# script rather than left to whatever the service currently has.
#
# WORKSPACE_SUBJECT's default here must track main.py's. It was left at
# ori@bdika.net when the code moved to exam@bdika.net on 2026-08-10, so every
# deploy that did not name it silently pointed the service back at a personal
# account — mail from the wrong address, uploads reached through a folder share
# instead of owned outright. A default that disagrees with the code's own is
# worse than no default: the code's is never reached, because this line always
# sets the variable.
gcloud run deploy "$SERVICE" \
    --source . \
    --function=process_form_responses \
    --project "$PROJECT" \
    --region "$REGION" \
    --no-allow-unauthenticated \
    --memory=2Gi \
    --set-secrets="GEMINI_API_KEY=gemini-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,GROQ_API_KEY=groq-api-key:latest,AZURE_OPENAI_API_KEY=azure-openai-api-key:latest" \
    --set-env-vars="^|^RESPONSES_SHEET_ID=${RESPONSES_SHEET_ID}|WORKSPACE_SUBJECT=${WORKSPACE_SUBJECT:-exam@bdika.net}|FALLBACK_RESULTS_FOLDER_ID=${FALLBACK_RESULTS_FOLDER_ID:-}|DEMO_CHECKER_ALLOWED_DOMAINS=${DEMO_CHECKER_ALLOWED_DOMAINS:-}|MAX_FILES_PER_RESPONSE=${MAX_FILES_PER_RESPONSE:-1}|MAX_PAGES_PER_RESPONSE=${MAX_PAGES_PER_RESPONSE:-4}|MAX_ROWS_PER_RUN=${MAX_ROWS_PER_RUN:-5}|AZURE_OPENAI_ENDPOINT=https://haskala-foundry-resource.openai.azure.com/openai/v1|AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini|LOG_LEVEL=${DEMO_CHECKER_LOG_LEVEL:-INFO}"

echo "✅ deployed."
echo "   First deploy only: run ./setup_infra.sh to wire the Cloud Scheduler trigger."
echo
echo "ℹ️  This form is open to anyone with the link — that is the product, not an"
echo "   oversight. What bounds the bill is MAX_PAGES_PER_RESPONSE=${MAX_PAGES_PER_RESPONSE:-4}"
echo "   (~\$0.05/page). Set DEMO_CHECKER_ALLOWED_DOMAINS only to close it down."
if [[ -z "${FALLBACK_RESULTS_FOLDER_ID:-}" ]]; then
    echo "⚠️  FALLBACK_RESULTS_FOLDER_ID is empty. Mail is the ONLY delivery path in"
    echo "   this product, so a graded report that cannot be sent is simply lost."
fi
