#!/bin/bash
# A contact sheet of film stills: ./sheet.sh name t1 t2 t3 t4  -> out/sheet_name.jpg (2x2)
name=$1; shift
python3 "$(dirname "$0")/peek.py" "$@" | grep -v duration
cd "$(dirname "$0")/out" && ffmpeg -hide_banner -loglevel error -y -i peek_$1.jpg -i peek_$2.jpg -i peek_$3.jpg -i peek_$4.jpg \
  -filter_complex "[0]scale=960:540[a];[1]scale=960:540[b];[2]scale=960:540[c];[3]scale=960:540[d];[a][b]hstack[t];[c][d]hstack[u];[t][u]vstack" -q:v 3 sheet_$name.jpg
