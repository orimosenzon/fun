#!/usr/bin/env bash
# פריסת שירות הבדיקה ל-Cloud Run בפרויקט ה-sandbox.
# הרצה: ./deploy.sh   (אחרי gcloud auth login פעם אחת — ראה README)
set -euo pipefail

PROJECT="oris-sandbox-501014"
REGION="me-west1"          # תל אביב — הקרוב ביותר
SERVICE="cloudrun-hello"

gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT" \
  --region "$REGION" \
  --allow-unauthenticated
