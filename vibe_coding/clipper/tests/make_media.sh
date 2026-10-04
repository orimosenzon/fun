#!/usr/bin/env bash
# יוצר קבצי בדיקה: סרטון מקור 16:9 עם שעון וצליל, פתיחה אדומה, סיום כחול ולוגו ירוק שקוף.
# הצבעים האחידים מאפשרים לבדוק בייצוא איזה חלק מופיע באיזו שנייה.
set -euo pipefail
cd "$(dirname "$0")/media"
q="-loglevel error -y"
ffmpeg $q -f lavfi -i "testsrc2=size=1280x720:rate=30:duration=30" -f lavfi -i "sine=frequency=440:duration=30" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest source.mp4
ffmpeg $q -f lavfi -i "color=c=0xd02020:size=1080x1920:rate=30:duration=2" -f lavfi -i "sine=frequency=880:duration=2" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest intro.mp4
ffmpeg $q -f lavfi -i "color=c=0x2040d0:size=1920x1080:rate=30:duration=2" -f lavfi -i "sine=frequency=220:duration=2" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest outro.mp4
ffmpeg $q -f lavfi -i "color=c=0x00ff00@1.0:size=400x200,format=rgba" -frames:v 1 logo.png
ls -la
