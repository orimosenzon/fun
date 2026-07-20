#!/usr/bin/env bash
# One-time Pub/Sub trigger wiring for oris-scanner. Run once, AFTER the first
# ./deploy.sh (needs the deployed Cloud Run URL). Safe to re-run (idempotent
# where possible) but not needed on every deploy.
#
# Unlike scan2's original setup, this bakes in the lessons from the retry-
# storm incident (2026-07-14 → 2026-07-20, ~585k requests, only 23 succeeded)
# from day one instead of adding them after the fact:
#   - ack deadline maxed out (600s) instead of the default 10s that guaranteed
#     every real scan would time out before finishing.
#   - fixed 600s backoff instead of immediate redelivery.
#   - dead-letter topic + max 5 delivery attempts, so a stuck message gives up
#     instead of looping forever.
# The Firestore dedup ledger in main.py is still the primary fix — this is
# the safety net for whatever it doesn't catch (e.g. genuinely new but
# perpetually-failing submissions).

set -euo pipefail

PROJECT="${ORIS_SCANNER_PROJECT:-master-gecko-500709-t0}"
REGION="${ORIS_SCANNER_REGION:-europe-west1}"
SERVICE="oris-scanner"
TOPIC="orisScanNow"
DEAD_LETTER_TOPIC="oris-scanner-dead-letter"
SUB="oris-scanner-sub"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
PUBSUB_SA="service-${PROJECT_NUMBER}@gcp-sa-pubsub.iam.gserviceaccount.com"
INVOKER_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

SERVICE_URL="$(gcloud run services describe "$SERVICE" --project="$PROJECT" --region="$REGION" --format='value(status.url)')"
echo "service URL: $SERVICE_URL"

echo "=== topics ==="
gcloud pubsub topics create "$TOPIC" --project="$PROJECT" 2>&1 || echo "  ($TOPIC already exists)"
gcloud pubsub topics create "$DEAD_LETTER_TOPIC" --project="$PROJECT" 2>&1 || echo "  ($DEAD_LETTER_TOPIC already exists)"
gcloud pubsub subscriptions create "${DEAD_LETTER_TOPIC}-sub" --topic="$DEAD_LETTER_TOPIC" --project="$PROJECT" 2>&1 || echo "  (dead-letter sub already exists)"

echo "=== IAM: Pub/Sub service agent may publish to dead-letter topic ==="
gcloud pubsub topics add-iam-policy-binding "$DEAD_LETTER_TOPIC" --project="$PROJECT" \
  --member="serviceAccount:${PUBSUB_SA}" --role="roles/pubsub.publisher"

echo "=== IAM: invoker service account may call the private Cloud Run service ==="
gcloud run services add-iam-policy-binding "$SERVICE" --project="$PROJECT" --region="$REGION" \
  --member="serviceAccount:${INVOKER_SA}" --role="roles/run.invoker"

echo "=== push subscription (hardened from day one) ==="
gcloud pubsub subscriptions create "$SUB" \
  --project="$PROJECT" \
  --topic="$TOPIC" \
  --push-endpoint="$SERVICE_URL" \
  --push-auth-service-account="$INVOKER_SA" \
  --ack-deadline=600 \
  --min-retry-delay=600s --max-retry-delay=600s \
  --dead-letter-topic="projects/${PROJECT}/topics/${DEAD_LETTER_TOPIC}" \
  --max-delivery-attempts=5 \
  2>&1 || echo "  ($SUB already exists — update manually if settings need to change)"

echo "=== IAM: Pub/Sub service agent may consume from $SUB (required for dead-lettering) ==="
gcloud pubsub subscriptions add-iam-policy-binding "$SUB" --project="$PROJECT" \
  --member="serviceAccount:${PUBSUB_SA}" --role="roles/pubsub.subscriber"

echo "✅ infra ready. Publish to '$TOPIC' to trigger a scan:"
echo "   gcloud pubsub topics publish $TOPIC --project=$PROJECT --message='go'"
