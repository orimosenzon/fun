#!/usr/bin/env bash
# One-time Cloud Scheduler trigger wiring for oris-scanner. Run once, AFTER the
# first ./deploy.sh (needs the deployed Cloud Run URL). Safe to re-run.
#
# oris-scanner previously went through Pub/Sub (topic + push subscription +
# dead-letter topic), inherited from scan2's design. That extra hop bought
# nothing here: we're polling on a schedule, not reacting to Classroom push
# notifications (the only case that requires Pub/Sub, since Classroom's
# Registrations API can only target a Pub/Sub topic, not a webhook — that
# path was explicitly deferred). Cloud Scheduler can call the Cloud Run HTTP
# endpoint directly, with its own retry/backoff config doing the same job
# Pub/Sub's ack-deadline/redelivery used to do — one retry system instead of
# two stacked ones.
#
# Polling frequency (measured 2026-07-27, one-minute polling):
#   - an idle scan takes ~6.0s and makes 15 Classroom API calls
#   - that is 2.5% of the binding Classroom quota (600 req/min/project/user;
#     the other two limits, 3000/min/project and 4M/day/project, are far away)
#   - Cloud Run cost ~$2/month: 262,980 vCPU-s/month against a 180,000 vCPU-s
#     free tier, so only the CPU overage is billed. Memory (131,490 GiB-s of
#     360,000) and requests (43,830 of 2M) stay free. The free tier is shared
#     with checker/scan1/scan2, so ~$6.7/month if they consume it first.
# Anything at */5 or slower fits entirely inside the free tier and costs $0,
# so */5 — not */15 — is the cheap option if the cost ever matters; 15 minutes
# buys nothing over 5. Kept at one minute for now: fast feedback matters while
# the Shamir pilot is still being demoed and debugged live.
#
# Retry settings below intentionally mirror the old Pub/Sub subscription
# config (ack deadline 600s, fixed 600s backoff, max 5 attempts) — same
# lessons from the retry-storm incident (2026-07-14 → 2026-07-20, ~585k
# requests, only 23 succeeded), just expressed as Scheduler's own knobs. The
# Firestore dedup ledger in main.py remains the primary safety net regardless
# of what triggers a scan.

set -euo pipefail

PROJECT="${ORIS_SCANNER_PROJECT:-master-gecko-500709-t0}"
REGION="${ORIS_SCANNER_REGION:-europe-west1}"
SERVICE="oris-scanner"
JOB="oris-scanner-poll"
SCHEDULE="${ORIS_SCANNER_SCHEDULE:-* * * * *}"  # every minute — see the cost note above
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
INVOKER_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

SERVICE_URL="$(gcloud run services describe "$SERVICE" --project="$PROJECT" --region="$REGION" --format='value(status.url)')"
echo "service URL: $SERVICE_URL"

echo "=== IAM: invoker service account may call the private Cloud Run service ==="
gcloud run services add-iam-policy-binding "$SERVICE" --project="$PROJECT" --region="$REGION" \
  --member="serviceAccount:${INVOKER_SA}" --role="roles/run.invoker"

echo "=== Cloud Scheduler job (polls every minute, hardened retry config) ==="
gcloud scheduler jobs create http "$JOB" \
  --project="$PROJECT" \
  --location="$REGION" \
  --schedule="$SCHEDULE" \
  --uri="$SERVICE_URL" \
  --http-method=POST \
  --oidc-service-account-email="$INVOKER_SA" \
  --oidc-token-audience="$SERVICE_URL" \
  --attempt-deadline=600s \
  --max-retry-attempts=5 \
  --min-backoff=600s --max-backoff=600s \
  2>&1 || echo "  ($JOB already exists — update manually if settings need to change, e.g.:)
  gcloud scheduler jobs update http $JOB --project=$PROJECT --location=$REGION --schedule='$SCHEDULE'"

echo "✅ infra ready. The job runs automatically on schedule ($SCHEDULE)."
echo "   To trigger a scan manually right now:"
echo "   gcloud scheduler jobs run $JOB --project=$PROJECT --location=$REGION"
echo "   To pause it (stop all triggering without deleting anything):"
echo "   gcloud scheduler jobs pause $JOB --project=$PROJECT --location=$REGION"
