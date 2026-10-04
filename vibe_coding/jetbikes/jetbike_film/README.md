# MJ-5: jet motorcycle, a test-flight film

A roughly two-minute animated film (1080p, 30 fps, with sound) of a rider on a flying jet motorcycle over an alpine valley. It is produced from a single physics simulation: nothing in the flight is animated by hand. The camera, the effects and the sound are all derived from the recording of that simulation.

- The film: `out/jetbike.mp4` (not in git, produced by the steps below)
- Interactive player: `python3 -m http.server` inside the folder, then `http://localhost:8000/`. It has scrubbing, the film's cameras, and a free camera.

## The vehicle

| Part | Details |
|---|---|
| Lift | 4 vertical turbojets, 1,300 N each at sea level, exhausting **downward** through nozzles with two-axis deflection vanes (±15°). The pairs spin in opposite directions so their gyroscopic moments cancel |
| Cruise | 1 turbojet with a normal tailpipe **pointing backward**, 1,600 N |
| Weight | 325 kg at takeoff: 82 kg rider, 62 kg fuel, about 20 kg per engine |
| Base altitude | 1,100 m, so air density is about 90% of sea level and so is the maximum thrust |

The numbers are in the range of real small turbojets (PBS TJ100 / JetCat class).

## The physics (`js/physics.js`)

- A rigid body in six degrees of freedom, attitude as a quaternion, integrated at 2,000 Hz.
- **Turbojet dynamics:** thrust ∝ rpm^3.2, a first-order spool lag of 0.2 s, and acceleration/deceleration limits like a real engine controller (FADEC). Start-up sequence: starter, ignition, acceleration to idle. On shutdown the rotor coasts down.
- **Intake momentum drag** (ram drag) for every engine, applied at its intake. It is a significant force in fast forward flight and also produces a pitching moment.
- Anisotropic body drag with separate centres of pressure, plus aerodynamic rate damping.
- **Gyroscopic coupling** of the spinning rotors, and the reaction torque when they spool up.
- **Ground effect:** jet suck-down and hot-gas re-ingestion.
- Air density from the standard atmosphere.
- Fuel burn changes the mass, the centre of gravity and the inertia tensor.
- Landing pads modelled as springs and dampers with Coulomb friction.
- Wind: a logarithmic boundary layer plus turbulence spread over a von Kármán spectrum (frozen field).

## Flight control (`js/control.js`)

The rider sits on the bike, but the vehicle is flown by a flight computer at 200 Hz working from noisy sensors (fly-by-wire). The control chain:

1. Position tracking: PID plus reference-acceleration feed-forward.
2. The required force is split: the cruise engine pushes forward, and the lift system supplies the rest. The direction of the lift vector sets the desired attitude, the same way a quadcopter works.
3. Attitude loop and angular-rate loop.

**The key point is the two-speed allocation.** A turbojet responds too slowly (about 0.2 s) to hold attitude by differential thrust alone. The first version oscillated until the bike flipped. So:

- differential engine thrust takes the low-passed torque demand;
- the vanes (30 ms servos) supply whatever torque the engines' **actual** thrust (measured from rpm) is missing at that instant;
- yaw comes from the vanes alone.

The route (`js/mission.js`) is a 3D spline with a speed profile limited by longitudinal acceleration, braking and turn acceleration, with smoothed terrain following:

- vertical takeoff from a concrete pad;
- about 160 km/h along the valley;
- a pass 5 m above the lake;
- a banked turn (about 30°, 1.15 g) around a rocky island;
- braking in a nose-up attitude;
- landing on a lakeside meadow.

## Rendering (`js/render/`)

- Three.js on the GPU (headless Chrome with Vulkan).
- Procedural terrain with baked sun shadowing, spruce and deciduous forest, a reflective lake.
- Grass that bends in the jet blast.
- Dust and water spray emitted where the exhaust hits the surface, using a jet velocity decay model U ≈ Ve·6D/h.
- Heat-haze distortion volumes behind the nozzles.
- Nozzle glow driven by exhaust gas temperature, vanes and compressors that move with the recording.

Post-processing: an HDR pipeline, bloom, and 8 sub-frames per video frame (a 180° shutter for motion blur, sub-pixel jitter for anti-aliasing), ACES tone mapping.

## Sound (`tools/audio.py`)

Synthesized from the telemetry:

- turbine whine at the shaft frequency,
- jet roar that rises with thrust,
- combustion rumble and ignition pops,
- the rush and splash of the jet on the ground or the water.

Every source moves with the bike, and each camera "hears" it at the retarded time t − r/c. That produces the Doppler effect and the propagation delay naturally. On top of that: 1/r spreading, air absorption of high frequencies, jet-noise directivity, stereo panning, crossfades at cuts, and birds in the quiet parts.

## Producing the film

```bash
python3 tools/render.py audio-data     # telemetry for the sound
python3 tools/audio.py                 # out/audio.wav
python3 tools/render.py frames         # out/frames/*.jpg (resumable; about 0.45 s per frame on a GTX 1060)
python3 tools/render.py encode         # out/jetbike.mp4
node tests/fly.mjs 5                   # flight telemetry in the terminal, without rendering
```
