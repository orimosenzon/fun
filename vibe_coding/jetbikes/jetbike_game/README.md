# MJ-5: the valley course (a game)

A browser flight game in the spirit of [fable](../fable/), built on the engine of the film [jetbike_film](../jetbike_film/):
- the same graphics (valley, lake, forest, heat haze, dust and spray, bloom, motion blur);
- the same physics, now flown in real time by a human.

**How to run:** `python3 -m http.server 8790` inside the folder, then `http://localhost:8790/`. There's no build step; Three.js loads from a CDN.

## The course

The course has 12 rings:
- along the valley,
- two low rings 4.5 m above the lake (the jet sprays the water),
- south of the rock island,
- through a gorge with walls about 500 m high,
- up to a high ring.

It ends with a **precision landing** in the circle on the far meadow. Time starts at liftoff. Landing close to the centre earns up to 5 seconds off, and the best time is saved in the browser.

## Controls

| Key | Action |
|---|---|
| `E` / `D` | Climb / descend (hands off holds altitude; below 40 m it holds height above the terrain) |
| `←` `→` | Bank, which produces a coordinated turn |
| `↑` `↓` | Nose down / up. At hover the bike moves forward/back; at speed it dives or climbs and trades speed for height. Holding `↓` hard is how you brake |
| `W` / `S` | Cruise engine throttle |
| `Z` / `X` | Rudder (yaw) |
| `Space` | Emergency power: +15% thrust, but the engine heats up and the engine computer cuts it at the limit |
| `C` | Camera: chase, far, rider's eyes, broadcast |
| `T` | Manual flight computer (the stick commands angular rates, `E`/`D` move a lift lever, no stabilization) |
| `R` / `M` / `H` | Back to the last checkpoint / mute / menu |

A gamepad works (left stick, right stick, triggers for throttle, A for emergency power), and there are basic touch controls on phones.

## What makes the physics believable

The bike is the film's rigid body: 5 turbojets, 2 kHz integration, drag, intake momentum drag, ground effect, wind, and fuel burn. The player never moves the bike directly. The input goes to a **flight computer** (`js/pilot.js`), and every force still passes through turbine spool lag, thrust limits and the vane servos. The differences you feel:

- **Turning costs thrust.** A bank of φ requires lift W/cos φ. The flight computer limits bank to what the engines can actually carry (about 42° at hover, up to 58° at full throttle), and pushing it makes the bike slowly sink and lose speed.
- **Vertical takeoff and landing like a real VTOL.** Engines start one after another, and there's an interlock: no liftoff until all four are at idle. Touching down harder than 5.5 m/s, or faster than 16 m/s horizontally, crashes the gear.
- **Braking with the nose up.** There's no brake: you tilt the lift vector backward. That's physically how a hoverbike decelerates.
- **Trading energy:** a dive speeds you up, pulling up slows you down.
- **A design fix discovered through the physics.** The cruise engine sits under the seat, so its thrust line passed 35 cm below the centre of gravity. That created a 500 N·m nose-up moment, which saturated the rear engines and made the bike sink in turns. The fix is a tailpipe bent 31.5° downward, with the angle solved numerically so the thrust line passes exactly through the centre of gravity. As a bonus, half of the cruise thrust helps carry the weight.
- **Floors that protect attitude control.** In a steep dive the flight computer won't drop collective below about 0.42 g, because without thrust there is no differential to control attitude with. That failure actually happened in testing, as a loss of control at 200 km/h.
- Wind with gusts, drift at hover, dust and spray only where the jet actually hits the surface, and noise from the jet on the ground.

## Crashes

Crashes are triggered by:
- the airframe or rider touching the ground,
- water,
- trees (a collision model of the crown and trunk),
- a ring's rim,
- a hard landing,
- running out of fuel.

Each ends in an explosion or a water spray, and after about 3.5 seconds you return to the last ring you passed.

## Code

| File | Contents |
|---|---|
| `js/physics.js`, `vehicle.js`, `terrain.js` | From the film. Changes: emergency power, reset, the bent tailpipe, the extended gorge, crash probe points |
| `js/fcs_core.js` | The film's allocator (slow differential thrust plus fast vanes) |
| `js/pilot.js` | The flight computer for a human pilot (assist and manual modes) |
| `js/course.js` | The rings and the landing target |
| `js/main.js` | Game loop, collisions, cameras, demo autopilot |
| `js/render/*` | The film's world with tree LOD, rings, explosion, post-processing with motion blur |
| `js/audio.js` | Live Web Audio sound: turbine whine, jet roar, wind, splash, Doppler |
| `js/input.js`, `hud.js` | Controls and a Hebrew HUD |

### Performance

- The detailed trees are only near the camera (400 m chunks).
- The lake reflection refreshes every other frame.
- The render resolution adapts to the frame rate.
- On a GTX 1060 at 1080p it runs at about 50–60 fps.

## Tests and tools

- `node tests/pilot.mjs` / `node tests/pilot_air.mjs`: scripted flights (takeoff, turn, dive, braking, hover) with telemetry.
- `python3 tools/play.py OUT --q demo`: the autopilot flies the course headless on the GPU, with screenshots and fps.
- `python3 tools/keys.py OUT`: real keyboard events.

Debug URL flags:
- `?demo`: autopilot;
- `?pose=final`: start near the landing target;
- `?cam=cockpit|far|tv`: starting camera;
- `?q=0.6`: quality;
- `?scale=0.7`: render resolution;
- `?touch`: touch controls.
