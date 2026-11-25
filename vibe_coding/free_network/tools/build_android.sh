#!/usr/bin/env bash
# Builds the Android app and puts it where the network offers it for download (/downloads/).
# The site picks it up on the next `npm run deploy`. Signing needs
# ~/.config/free-network/android-signing.properties and the key it names (outside the repo).
set -euo pipefail
cd "$(dirname "$0")/../android"
[ -f ~/.config/free-network/android-signing.properties ] || { echo "Missing the signing key in ~/.config/free-network/" >&2; exit 1; }
[ -f local.properties ] || echo "sdk.dir=${ANDROID_HOME:-$HOME/android-sdk}" > local.properties
./gradlew assembleRelease --console=plain -q
cp app/build/outputs/apk/release/app-release.apk ../public/downloads/free-network.apk
ls -la ../public/downloads/free-network.apk
