#!/bin/bash
W=$HOME/.cache/pardespedia_interviews
IDS="oNHE5PFHxgY jEhM7AYHXj0 BIOpgT2fqI8 Jlwmd-pTtoI ft5Ke0L1pMc"
cd $W/audio
for id in $IDS; do
  [ -f $id.m4a ] || yt-dlp -q --no-warnings -f bestaudio -x --audio-format m4a -o "$id.%(ext)s" "https://www.youtube.com/watch?v=$id" 2>&1 | grep -v Deprecated
  [ -f $id.m4a ] && echo "DL ok $id" || echo "DL FAILED $id"
done
cd $W/bin
systemd-run --user --scope -q -p MemoryMax=7G python3 $W/bin/transcribe5.py $IDS || echo "TR FAILED"
systemd-run --user --scope -q -p MemoryMax=7G /home/ori/fun/vibe_coding/transparency/.venv/bin/python $W/bin/diarize5.py $IDS || echo "DZ FAILED"
echo PIPELINE END
