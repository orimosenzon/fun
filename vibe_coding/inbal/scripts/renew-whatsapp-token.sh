#!/bin/bash
# Usage: ./scripts/renew-whatsapp-token.sh NEW_TOKEN
# Run this every ~2 months after generating a new token from:
# https://developers.facebook.com → inbal-ceramics → WhatsApp → API Setup → Generate access token

set -e

NEW_TOKEN=$1

if [ -z "$NEW_TOKEN" ]; then
  echo "Usage: ./scripts/renew-whatsapp-token.sh NEW_TOKEN"
  echo ""
  echo "Get a new token from:"
  echo "  https://developers.facebook.com → inbal-ceramics → WhatsApp → API Setup → Generate access token"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env.local"

# Update .env.local
sed -i "s|WHATSAPP_ACCESS_TOKEN=.*|WHATSAPP_ACCESS_TOKEN=$NEW_TOKEN|" "$ENV_FILE"
echo "✓ Updated .env.local"

# Verify the token works
echo "Verifying token..."
RESULT=$(curl -s -X POST \
  "https://graph.facebook.com/v21.0/968039886402071/messages" \
  -H "Authorization: Bearer $NEW_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messaging_product":"whatsapp","to":"972508820997","type":"template","template":{"name":"standby_notification","language":{"code":"he"},"components":[{"type":"body","parameters":[{"type":"text","text":"בדיקה"}]}]}}')

if echo "$RESULT" | grep -q '"messages"'; then
  echo "✓ Token works — test message sent to Ori"
else
  echo "✗ Token verification failed:"
  echo "$RESULT"
  exit 1
fi

echo ""
echo "Done! Now update the token in Vercel:"
echo "  1. Go to https://vercel.com/dashboard → inbal → Settings → Environment Variables"
echo "  2. Find WHATSAPP_ACCESS_TOKEN and update it"
echo "  3. Redeploy"
