#!/bin/bash
# Deploys the private, keyed copy of the game to Cloudflare (archon.<account>.workers.dev).
# The secret key lives in Cloudflare (wrangler secret put KEY) and in deploy/.key, never in git.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf public && mkdir -p public/data
cp -r ../index.html ../src public/
cp ../data/kernal.bin ../data/basic.bin ../data/chargen.bin ../data/archon.d64 public/data/
npx --yes wrangler@4 deploy
