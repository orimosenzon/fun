#!/usr/bin/env bash
# Render every scene of a ManimGL file and concatenate them into one video.
# Usage: ./render.sh [--hd|--uhd|-l] [file.py]   (default: --hd scenes/zeta_primes.py)
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

QUALITY="${1:---hd}"
FILE="${2:-scenes/zeta_primes.py}"
NAME="$(basename "$FILE" .py)"
OUT_DIR="videos/$NAME"
mkdir -p "$OUT_DIR"

SCENES=$(.venv/bin/python -c "
import sys, runpy; sys.argv=['x']
mod = runpy.run_path('$FILE')
print(' '.join(s.__name__ for s in mod['SCENES']))")

# ONLY="SceneA SceneB" ./render.sh  renders just those; the concat still uses every scene
for scene in ${ONLY:-$SCENES}; do
    echo "== $scene"
    .venv/bin/manimgl "$FILE" "$scene" -w "$QUALITY" --video_dir "$OUT_DIR" > "$OUT_DIR/$scene.log" 2>&1 \
        || { echo "FAILED: see $OUT_DIR/$scene.log"; tail -5 "$OUT_DIR/$scene.log"; exit 1; }
done

: > "$OUT_DIR/list.txt"
for scene in $SCENES; do echo "file '$scene.mp4'" >> "$OUT_DIR/list.txt"; done

# the narration of each scene ends before its fade-out; aresample fills those gaps with real silence
ffmpeg -loglevel error -y -f concat -safe 0 -i "$OUT_DIR/list.txt" \
    -c:v copy -af aresample=async=1 -c:a aac -b:a 192k -movflags +faststart "videos/$NAME.mp4"
echo "done: videos/$NAME.mp4"
