#!/usr/bin/env bash
# Read form-checker's Cloud Run logs — the trace of what happened when a
# teacher submitted the form.
#
# The service logs two levels. INFO is the story of a response actually being
# graded (claimed → checking parameters → per-page OCR → emailed), and stays
# quiet the rest of the time. DEBUG is the every-5-minutes background chatter
# (polls that found nothing, rows already graded and skipped by the ledger).
# DEBUG is only emitted at all if the service runs with LOG_LEVEL=DEBUG —
# see "enabling debug" at the bottom.
#
# Usage:
#   ./logs.sh                 # last 2h of INFO+ , oldest first
#   ./logs.sh -f              # follow (tail) live
#   ./logs.sh -n 300          # last 300 entries instead of the default 100
#   ./logs.sh -s 30m          # look back 30 minutes instead of 2h
#   ./logs.sh -d              # include DEBUG (the polling chatter)
#   ./logs.sh -g 'a1b2c3d4'   # only lines matching this text — e.g. the 8-char
#                             #   row key, to isolate one response's trace
#   ./logs.sh -e              # errors and warnings only

set -euo pipefail

PROJECT="${FORM_CHECKER_PROJECT:-master-gecko-500709-t0}"
REGION="${FORM_CHECKER_REGION:-europe-west1}"
SERVICE="form-checker"

# The SDK is installed under $HOME rather than system-wide, and its PATH entry
# only exists in interactive shells that source the profile.
if ! command -v gcloud >/dev/null 2>&1 && [[ -x "$HOME/google-cloud-sdk/bin/gcloud" ]]; then
    PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi
if ! command -v gcloud >/dev/null 2>&1; then
    echo "❌ gcloud not found on PATH (looked in \$HOME/google-cloud-sdk/bin too)" >&2
    exit 1
fi

# "2h" / "30m" / "3d" → the absolute UTC timestamp Cloud Logging wants.
# GNU date rejects the shorthand units, hence the spelling-out.
_since_to_timestamp() {
    local spec="$1" n unit word
    n="${spec%%[a-zA-Z]*}"
    unit="${spec##*[0-9]}"
    case "$unit" in
        h|H) word=hours ;;
        m|M) word=minutes ;;
        d|D) word=days ;;
        *) echo "❌ bad -s value '$spec' — use e.g. 30m, 2h, 3d" >&2; exit 1 ;;
    esac
    date -u -d "$n $word ago" +%Y-%m-%dT%H:%M:%SZ
}

FOLLOW=0
LIMIT=100
SINCE="2h"
MIN_SEVERITY="INFO"
GREP=""

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while getopts "fn:s:dg:eh" opt; do
    case "$opt" in
        f) FOLLOW=1 ;;
        n) LIMIT="$OPTARG" ;;
        s) SINCE="$OPTARG" ;;
        d) MIN_SEVERITY="DEBUG" ;;
        g) GREP="$OPTARG" ;;
        e) MIN_SEVERITY="WARNING" ;;
        h) usage 0 ;;
        *) usage 1 ;;
    esac
done

# Restricted to the app's own stdout/stderr: cloud_run_revision also carries
# run.googleapis.com/requests, one access-log entry per poll with no message
# body at all, which on a polling schedule is pure noise.
FILTER="resource.type=cloud_run_revision
resource.labels.service_name=${SERVICE}
resource.labels.location=${REGION}
logName=(\"projects/${PROJECT}/logs/run.googleapis.com%2Fstdout\" OR \"projects/${PROJECT}/logs/run.googleapis.com%2Fstderr\")
severity>=${MIN_SEVERITY}"

if [[ -n "$GREP" ]]; then
    # textPayload for plain log lines, jsonPayload.message for structured ones
    FILTER+="
(textPayload:\"${GREP}\" OR jsonPayload.message:\"${GREP}\")"
fi

if [[ "$FOLLOW" == 1 ]]; then
    echo "📡 following ${SERVICE} logs (severity>=${MIN_SEVERITY})… Ctrl-C to stop" >&2
    exec gcloud beta logging tail "$FILTER" \
        --project "$PROJECT" \
        --format='value(timestamp.date("%H:%M:%S"), severity, textPayload, jsonPayload.message)'
fi

echo "📜 last ${LIMIT} entries from the past ${SINCE} (severity>=${MIN_SEVERITY})" >&2
gcloud logging read "${FILTER}
timestamp>=\"$(_since_to_timestamp "$SINCE")\"" \
    --project "$PROJECT" \
    --limit "$LIMIT" \
    --order desc \
    --format='value(timestamp.date("%m-%d %H:%M:%S"), severity, textPayload, jsonPayload.message)' \
    | tac

# ─── enabling debug ─────────────────────────────────────────────────────────
# The service defaults to INFO, so -d shows nothing extra until DEBUG is
# turned on server-side. Redeploy with it on:
#   FORM_CHECKER_LOG_LEVEL=DEBUG ./deploy.sh "temporarily verbose logging"
# and plain ./deploy.sh puts it back to INFO.
#
# For a quick look without a redeploy:
#   gcloud run services update form-checker --region europe-west1 \
#       --update-env-vars LOG_LEVEL=DEBUG
# (the next ./deploy.sh resets it — deploy.sh sets the env block wholesale.)
