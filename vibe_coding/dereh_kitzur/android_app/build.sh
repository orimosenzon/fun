#!/usr/bin/env bash
# Build the Android app.
#
#   ./build.sh            the real thing: loads the live site from GitHub Pages
#   ./build.sh dev        loads http://127.0.0.1:8790/index.html instead, for
#                         trying web changes on a phone or the emulator before
#                         they are pushed (127.0.0.1 and not localhost: Capacitor
#                         keeps that name for itself). Serve web/ there and
#                         forward the port:
#                           python3 -m http.server 8790 -d ../web
#                           adb reverse tcp:8790 tcp:8790
#
# The APK lands in android/app/build/outputs/apk/debug/app-debug.apk.
set -euo pipefail
cd "$(dirname "$0")"

export JAVA_HOME="${JAVA_HOME_21:-$(ls -d "$HOME"/.jdks/jdk-21* | head -1)}"
npx cap sync android >/dev/null

if [ "${1:-}" = "dev" ]; then
  # Only the copy inside the APK is changed; capacitor.config.json stays the
  # real one, and the next plain build puts the live URL back.
  python3 - <<'EOF'
import json, pathlib
p = pathlib.Path('android/app/src/main/assets/capacitor.config.json')
c = json.loads(p.read_text())
c['server']['url'] = 'http://127.0.0.1:8790/index.html?src=app'
c['server']['cleartext'] = True
p.write_text(json.dumps(c, ensure_ascii=False, indent=2))
EOF
  echo "dev build: loads http://127.0.0.1:8790"
fi

(cd android && ./gradlew assembleDebug -q 2>&1 | grep -v '^Note:' || true)
ls -la android/app/build/outputs/apk/debug/app-debug.apk
