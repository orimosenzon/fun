#!/usr/bin/env bash
# One-time Cloud Scheduler trigger wiring for form-checker. Run once, AFTER the
# first ./deploy.sh (it needs the deployed Cloud Run URL). Safe to re-run.
#
# Same shape as oris-scanner's: Cloud Scheduler calls the private Cloud Run HTTP
# endpoint directly, with an OIDC token. No Pub/Sub — see oris-scanner's
# setup_infra.sh for why that hop was removed (one retry system instead of two
# stacked ones, after a retry storm produced ~585k requests over six days).
#
# POLLING FREQUENCY — */5, and here the reasoning differs from oris-scanner's.
#
# oris-scanner runs at */1 because a class hands in together and a teacher is
# watching for the reports. A form response has no such audience: the teacher
# submitted and closed the tab, and the report arrives by email whenever it
# arrives. Five minutes is invisible to them.
#
# It is also the free option. Using oris-scanner's measured figures (an idle
# scan ~6s), per month at */5: ~8,766 requests, ~52,600 vCPU-s, ~26,300 GiB-s —
# all inside the Cloud Run free tier (2M requests, 180,000 vCPU-s, 360,000
# GiB-s). At */1 it would be ~263,000 vCPU-s, i.e. ~$2/month of CPU overage.
# Caveat: that free tier is shared with oris-scanner and the other services in
# this project, so if they consume it first this is ~$1.3/month rather than $0.
#
# Retry settings mirror oris-scanner's (600s attempt deadline, fixed 600s
# backoff, max 5 attempts). The Firestore ledger in main.py remains the primary
# safety net regardless of what triggers a scan — and note MAX_ROWS_PER_RUN
# caps a single invocation, so a backlog drains over several polls instead of
# one run overrunning the attempt deadline and being retried from the top.

set -euo pipefail

PROJECT="${FORM_CHECKER_PROJECT:-master-gecko-500709-t0}"
REGION="${FORM_CHECKER_REGION:-europe-west1}"
SERVICE="math-form-checker"
JOB="math-form-checker-poll"
SCHEDULE="${FORM_CHECKER_SCHEDULE:-*/5 * * * *}"  # see the cost note above
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
INVOKER_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

SERVICE_URL="$(gcloud run services describe "$SERVICE" --project="$PROJECT" --region="$REGION" --format='value(status.url)')"
echo "service URL: $SERVICE_URL"

echo "=== IAM: invoker service account may call the private Cloud Run service ==="
gcloud run services add-iam-policy-binding "$SERVICE" --project="$PROJECT" --region="$REGION" \
  --member="serviceAccount:${INVOKER_SA}" --role="roles/run.invoker"

echo "=== Cloud Scheduler job (polls every 5 minutes, hardened retry config) ==="
gcloud scheduler jobs create http "$JOB" \
  --project="$PROJECT" \
  --location="$REGION" \
  --schedule="$SCHEDULE" \
  --uri="$SERVICE_URL" \
  --http-method=POST \
  --oidc-service-account-email="$INVOKER_SA" \
  --oidc-token-audience="$SERVICE_URL" \
  --attempt-deadline=1800s \
  --max-retry-attempts=5 \
  --min-backoff=600s --max-backoff=600s \
  2>&1 || echo "  ($JOB already exists — update manually if settings need to change, e.g.:)
  gcloud scheduler jobs update http $JOB --project=$PROJECT --location=$REGION --schedule='$SCHEDULE'"

echo "✅ infra ready. The job runs automatically on schedule ($SCHEDULE)."
echo "   To trigger a scan manually right now:"
echo "   gcloud scheduler jobs run $JOB --project=$PROJECT --location=$REGION"
echo "   To pause it (stop all triggering without deleting anything):"
echo "   gcloud scheduler jobs pause $JOB --project=$PROJECT --location=$REGION"
echo
echo "REMINDER — the two scopes this service needs on the domain-wide-delegation"
echo "entry (Workspace admin console, client id 114647217076557059736):"
echo "   https://www.googleapis.com/auth/spreadsheets.readonly   (required)"
echo "   https://www.googleapis.com/auth/gmail.send              (delivery; degrades to Drive)"
echo "Adding scopes there is additive and cannot affect oris-scanner."
