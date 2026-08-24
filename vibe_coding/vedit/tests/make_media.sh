#!/usr/bin/env bash
# יוצר את קובצי הבדיקה. דורש ffmpeg.
set -e
cd "$(dirname "$0")"
mkdir -p testmedia

ffmpeg -y -loglevel error -f lavfi -i "testsrc=duration=6:size=640x360:rate=30" \
  -f lavfi -i "sine=frequency=440:duration=6" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest testmedia/clip_a.mp4

ffmpeg -y -loglevel error -f lavfi -i "smptebars=duration=5:size=640x360:rate=30" \
  -f lavfi -i "sine=frequency=880:duration=5" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest testmedia/clip_b.mp4

ffmpeg -y -loglevel error \
  -f lavfi -i "color=c=0x1e6fa8:size=640x360:duration=5:rate=30,drawtext=text='SCENE C':fontsize=90:fontcolor=white:x=(w-tw)/2:y=(h-th)/2" \
  -f lavfi -i "sine=frequency=220:duration=5" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest testmedia/clip_c.mp4

ffmpeg -y -loglevel error -f lavfi -i "testsrc=duration=10:size=1280x720:rate=30" \
  -f lavfi -i "sine=frequency=440:duration=10" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest testmedia/real720.mp4

ffmpeg -y -loglevel error -f lavfi -i "sine=frequency=440:duration=8" \
  -c:a libmp3lame testmedia/music.mp3

ffmpeg -y -loglevel error -f lavfi -i "testsrc=size=800x600:duration=1:rate=1" \
  -frames:v 1 testmedia/photo.png

echo "נוצרו קובצי בדיקה ב-testmedia/"
