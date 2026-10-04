# jetbike_game: MJ-5 game, the valley course

## What it is
A browser flight game (in the spirit of fable) built on the jetbike_film engine: the same graphics and physics, in real time. 12 rings plus a precision landing. Hebrew UI.
Started 28/9/2026 as a follow-up to the film (jetbike_film).

## Stack
Plain ES modules plus Three.js 0.169 from jsdelivr. No build. Run with `python3 -m http.server` → index.html.

## Key decisions
- **The flight computer (pilot.js) translates input into commands for the engines.** The player never moves the bike directly.
  - Assist mode: bank, then a coordinated turn; pitch follows the nose; altitude hold.
  - Radar-altitude hold below 40 m, looking about 1 s ahead. It was added after a newbie test crashed at 4 m.
  - Manual mode: rates plus a lift lever.
- **Bent tailpipe at 31.47°** so the cruise engine's thrust line passes through the centre of gravity. Without it, the rear lift engines saturated in turns. MAIN_TMAX was raised to 1900.
- Dynamic bank limit = acos((W − main-engine lift)/(0.85·max lift)). Minimum collective in the air is 0.42 W (keeps attitude control in a dive).
- Takeoff interlock until all four lift engines are running (the staggered start caused tipping).
- Trees: 400 m LOD chunks, and a 26 m corridor between rings is cleared of trees. Water reflection every other frame. Adaptive resolution.
- The tailpipe tip is not a crash probe (it's 23 cm above the ground).

## Tests
- `tests/pilot.mjs` and `tests/pilot_air.mjs`: scripted Node tests.
- `tools/play.py`: GPU demo run.
- `tools/keys.py`: real keyboard.
- Results on 28/9: the autopilot passes all 12 rings, and the finish via `?pose=final` works.

- **Loading screen** (index.html, `window.loader`): shown on first paint. buildWorld/gridMesh/makeGrass are async and yield every 50 ms through `makeSlicer` (stage weights were calibrated from measured times). The first render and shader warm-up happen behind the loader.

## Open for next time
- Nobody has flown it by hand yet: tuning the feel (gains in `ASSIST`) according to Ori's impressions.
- Touch controls are basic and untested on a real phone.
- The demo autopilot crashes into trees south of the rock (fine as a demo, but could be improved).
- Loading takes ~12 s (6 s of it is `heightAt` for the terrain grids). Candidates: a Web Worker, or a precomputed height grid.
