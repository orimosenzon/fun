#!/usr/bin/env bash
# Build the PC host and run the headless physics check.
set -e
cd "$(dirname "$0")"
# the ground photographs of the web build, baked for the console (needs python3 with Pillow)
if [ ! -f assets/ground.bin ] || [ tools/bake_ground.py -nt assets/ground.bin ]; then
  python3 tools/bake_ground.py >/dev/null || echo "note: could not bake assets/ground.bin, the ground will be drawn without photographs"
fi
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j >/dev/null
echo "built: build/foehn, build/test_sim"
echo "--- physics check ---"
./build/test_sim | tail -3
echo
echo "run interactively:  ./build/foehn"
echo "render a still:     ./build/foehn --shot out.ppm"
