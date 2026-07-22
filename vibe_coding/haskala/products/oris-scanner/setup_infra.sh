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
SCHEDULE="${ORIS_SCANNER_SCHEDULE:-*/15 * * * *}"  # every 15 minutes
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
INVOKER_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

SERVICE_URL="$(gcloud run services describe "$SERVICE" --project="$PROJECT" --region="$REGION" --format='value(status.url)')"
echo "service URL: $SERVICE_URL"

echo "=== IAM: invoker service account may call the private Cloud Run service ==="
gcloud run services add-iam-policy-binding "$SERVICE" --project="$PROJECT" --region="$REGION" \
  --member="serviceAccount:${INVOKER_SA}" --role="roles/run.invoker"

echo "=== Cloud Scheduler job (polls every 15 min, hardened retry config) ==="
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
