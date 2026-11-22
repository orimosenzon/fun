#!/usr/bin/env bash
# Run FÖHN on the PC host. Builds first when something has changed (a second or so when nothing has).
#
#   ./run.sh                      fly (afternoon, clear)
#   ./run.sh dusk                 the hour: morning, noon, afternoon, golden, dusk
#   ./run.sh golden rain          the weather: clear, rain, snow
#   ./run.sh calm | gale          the wind: calm, breeze, wind, gale
#   ./run.sh shot [file.png] ...  a still instead of a window (default foehn.png, in the folder you are in)
#   ./run.sh ... --invert         anything else goes to the program as it is
#                                 (--invert, --nopost, --shadow 0, --aniso 4, --pos x,z,height,yaw,speed, --side ...)
#
# Works from any folder. Keys, once it is up: left/right turn, DOWN climbs, UP dives, W/S throttle,
# A/D side-step, Space overdrive, Shift or B brake, R respawn, T hour, X weather, N wind, Esc quit.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
from="$PWD"

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi

cd "$here"
# the ground photographs, baked once (needs python3 with Pillow; without them the ground is only painted)
if [ ! -f assets/ground.bin ] || [ tools/bake_ground.py -nt assets/ground.bin ]; then
  python3 tools/bake_ground.py >/dev/null || echo "note: could not bake assets/ground.bin, the ground will be drawn without photographs"
fi
[ -f build/Makefile ] || [ -f build/build.ninja ] || cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
# quiet when it works; the compiler's own words when it does not
if ! log="$(cmake --build build -j --target foehn 2>&1)"; then echo "$log" | grep -E "error|Error" -A3 | head -40; echo "build failed"; exit 1; fi

shot=""; args=()
while [ $# -gt 0 ]; do
  case "$1" in
    morning|noon|afternoon|golden|dusk) args+=(--time "$1") ;;
    clear|rain|snow) args+=(--wx "$1") ;;
    calm) args+=(--wind 0) ;; breeze) args+=(--wind 1) ;; wind) args+=(--wind 2) ;; gale) args+=(--wind 3) ;;
    shot)
      shot="foehn.png"
      case "$2" in *.png) shot="$2"; shift ;; esac ;;
    *) args+=("$1") ;;
  esac
  shift
done

if [ -n "$shot" ]; then
  case "$shot" in /*) out="$shot" ;; *) out="$from/$shot" ;; esac
  tmp="$(mktemp --suffix=.ppm)"; trap 'rm -f "$tmp"' EXIT
  ./build/foehn --shot "$tmp" "${args[@]}" | tail -1 | sed 's/^wrote [^ ]* //'
  if command -v ffmpeg >/dev/null; then ffmpeg -y -loglevel error -i "$tmp" "$out" && echo "saved $out"
  else cp "$tmp" "${out%.png}.ppm" && echo "saved ${out%.png}.ppm (no ffmpeg here to make a PNG of it)"; fi
else
  exec ./build/foehn "${args[@]}"
fi
