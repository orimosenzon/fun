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
# Polling frequency (measured 2026-07-27; moved from */1 to */5 on 2026-07-28
# to bring the bill to zero, then back to */1 on 2026-07-30 at Ori's request —
# responsiveness over the ~$2/month of CPU overage; see the numbers below). An
# idle scan takes ~6.0s and makes 15 Classroom API calls, so per month at */5:
#   - 8,766 requests, ~52,600 vCPU-s, ~26,300 GiB-s
#   - all three inside the Cloud Run free tier (2M requests, 180,000 vCPU-s,
#     360,000 GiB-s) => $0. At */1 it was ~263,000 vCPU-s, i.e. ~$2/month of
#     CPU overage. Caveat: that free tier is shared with checker/scan1/scan2,
#     so if they consume it first this costs ~$1.3/month rather than nothing.
#   - Classroom quota is a non-issue either way: 15 calls per scan against
#     600/min/project/user (the binding limit; 3000/min/project and
#     4M/day/project are far away).
# */15 buys nothing over */5 — both are free — it only triples the wait, which
# is why 5 minutes is the floor worth taking.
#
# A point that argued for the shorter interval: a scan that dies partway
# (Classroom 503s do happen — three on 2026-07-28) is only recovered by the
# next poll — the whole scan is lost, including courses it had not reached yet.
# At */1 that gap is a minute; at */5 it was five. There is no per-call retry
# inside the scan;
# Scheduler's own retry is configured below but its 600s backoff means the
# next scheduled run almost always beats it.
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
SCHEDULE="${ORIS_SCANNER_SCHEDULE:-*/1 * * * *}"  # every minute — see the cost note above (~$2/month CPU overage)
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
