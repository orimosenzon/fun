#!/bin/bash
# Deploys the private, keyed copy of the game to Cloudflare (mule.<account>.workers.dev).
# The secret key lives in Cloudflare (wrangler secret put KEY) and in deploy/.key, never in git.
# The emulator is imported from ../archon/src, so the site keeps the repo layout: /mule/ and /archon/src/.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf public && mkdir -p public/mule/data public/archon
cp -r ../index.html ../how.html ../src ../guide public/mule/
cp -r ../../archon/src public/archon/
cp ../data/kernal.bin ../data/basic.bin ../data/chargen.bin ../data/mule.d64 public/mule/data/
npx --yes wrangler@4 deploy
