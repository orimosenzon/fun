/* מצב תעופה - flying over the moshava to find its shortcuts.
 *
 * The rest of the app answers "where is the shortcut I already know about".
 * This answers the other question: what is out there. You take off over the
 * satellite image, steer with the mouse, and trails light up as you come near
 * them - nothing is drawn until you are close enough to have found it. The
 * photos that are attached to a trail hang in the air above it and grow as
 * you close in, so a picture is a thing you fly towards.
 *
 * Built on the map that is already here rather than on a 3D engine. MapLibre
 * already holds the satellite raster, the terrain, and a camera that tilts to
 * 80 degrees; what it does not have is a flight model. So the whole of this
 * file is really three things: a camera that thinks in metres above the
 * ground, a reveal rule that decides what you have discovered, and an engine
 * you can hear.
 *
 * The camera is honest. `alt` is a real height in metres, and the zoom handed
 * to the map is derived from it through MapLibre's own projection geometry
 * (see `zoomFor`). That is what makes the photos grow correctly on approach -
 * they are sized in world metres and projected, not scaled by a fudge factor.
 *
 * The controls are an aircraft's, loosely. The mouse is the stick: where the
 * cursor sits left or right of the middle of the screen is how hard you turn,
 * no button held. The button is the throttle - held, the engine spools up;
 * released, it winds down. W and S are the elevator, nose down and nose up.
 * A and D roll, all the way round if you keep them down, and a bank turns you
 * the way a bank does. The point of all of it is to be able to see a photo in
 * the distance and simply go there.
 */
'use strict';

const Explore = (() => {

  /* ---------- tuning ----------
   *
   * Speeds and ranges scale with altitude, all of them. Ground covered per
   * second has to grow with height or climbing feels like slowing down, and
   * the reveal radius has to grow with it or climbing shows you a wider view
   * of nothing. Height is the one control that changes the game: low is a
   * walk-through, high is a survey. */

  const ALT_MIN = 45;
  const ALT_MAX = 1400;
  const ALT_START = 260;

  const PITCH_LOW = 80;      // skimming: a lot of horizon, dramatic
  const PITCH_HIGH = 74;     // high up: looking down, a sliver of sky
  const PITCH_MIN = 58;      // limits for the nose-up / nose-down trim
  const PITCH_MAX = 80;
  const TRIM_DOWN = 18;      // degrees of nose-down the mouse can ask for
  const TRIM_UP = 8;         // and of nose-up; the auto pitch already sits near the top

  /* Cruise is metres per second per metre of altitude, and the first numbers
   * here were three times too fast: 421 km/h at 260 m, which crosses the whole
   * four kilometres of the moshava in ten seconds. You were over open fields
   * before anything had a chance to light up.
   *
   * It survived testing because of the clamp on `dt` below. A headless browser
   * drawing five frames a second advances the model by 0.06 s per frame, so
   * every test flight ran at about a third of real time and felt reasonable.
   * A flight model has to be checked against the clock, not against how far it
   * got in a test. */
  const SPEED_K = 0.12;      // cruise m/s per metre of altitude
  const SPEED_MIN = 10;      // 36 km/h skimming the rooftops
  const SPEED_MAX = 65;      // 234 km/h at survey height
  const BOOST = 2.1;         // top speed, as a multiple of cruise: the afterburner band
  const THR_UP = 2.0;        // seconds of button held from idle to full throttle
  const THR_DOWN = 3.5;      // seconds from full throttle back to idle once released
  const ACCEL = 2.2;         // how fast the speed follows the throttle, per second

  const YAW_RATE = 70;       // degrees per second with the mouse at the edge
  const YAW_ACCEL = 5.0;
  const DEAD = 0.09;         // the middle of the screen where the mouse asks nothing
  const PIXEL_CAP = 1.5;     // device pixels per CSS pixel the map is allowed while flying

  const ROLL_RATE = 150;     // degrees per second with A or D held: a full roll in 2.4 s
  const ROLL_ACCEL = 6;
  const BANK_MAX = 22;       // how far a plain mouse turn banks the picture
  const BANK_TURN = 48;      // degrees per second of turn that a 90-degree bank buys
  const LEVEL_K = 2.6;       // how quickly the wings level once A and D are released
  const SINK_K = 0.08;       // banked over, the wings hold less: fraction of altitude lost per second, inverted

  const CLIMB_K = 0.85;      // fraction of current altitude gained per second

  const REVEAL_K = 5.0;      // reveal radius = altitude * this
  const REVEAL_MIN = 420;
  const REVEAL_MAX = 3000;
  const PHOTO_K = 4.0;
  const PHOTO_MIN = 340;
  const PHOTO_MAX = 2400;

  const MAX_CARDS = 8;       // floating photos on screen at once
  const MAX_CHIPS = 5;       // trail names on screen at once
  const MAX_LINES = 40;      // trails fed to the glow source

  const CARD_MIN_PX = 44;
  const CARD_MAX_VH = 0.42;
  const CHIP_MIN_PX = 11;
  const CHIP_MAX_PX = 30;

  const HOVER_MS = 120;      // how long the cursor rests on a photo before the stick lets go
  const BAND_H = 600;        // height of the sky gradient, in css; must match .fly-band

  /* The vertical field of view the viewer actually sees, in degrees. The map
   * container is larger than the screen (see `fitFrame`), so the field of
   * view handed to MapLibre is wider than this; what is held constant is the
   * picture in the window. */
  const VFOV = 32;

  const RAD = Math.PI / 180;
  const M_PER_DEG_LAT = 111320;

  /* ---------- state ---------- */

  let on = false;
  let flying = false;        // false during the take-off ease, true after
  let raf = null;
  let last = 0;
  let tick = 0;              // seconds until the next reveal recompute
  let frame = 0;

  const keys = new Set();
  let pos = { lat: 0, lng: 0 };
  let alt = ALT_START;
  let bearing = 0;
  let speed = 0;
  let throttle = 0;          // 0..1, driven by the mouse button
  let hold = false;          // the mouse button is down over open ground
  let yaw = 0;               // current turn rate, degrees/second
  let bank = 0;              // right wing down is positive; the picture is rotated by minus this
  let bankRate = 0;
  let pitchTrim = 0;         // nose up or down, from the mouse, on top of the auto pitch
  let burn = 0;              // afterburner, 0..1, eased for the sound and the glow

  const mouse = { nx: 0, ny: 0, in: false };   // stick position, -1..1 from the middle
  const hover = { hud: false, card: null, since: 0 };
  /* The stick is taken, not simply held. Whenever the mouse has been busy
   * being a mouse - reading the help, closing a picture, coming back into the
   * window - it is wherever that left it, and a stick that engages there
   * throws the aircraft into a turn nobody asked for. It engages again only
   * when the cursor passes through the ring in the middle of the screen. */
  let armed = false;

  let canRoll = false;       // fine pointer: a keyboard and a mouse are here
  let ox = 0, oy = 0;        // how far the map hangs past the screen, pixels
  let focal = 1.5;           // camera distance in container heights, see fitFrame

  let world = [];            // every item near enough to ever matter
  let shots = [];            // every photo, placed on the ground
  let seen = new Map();      // item id -> when it came into range, for the pulse
  let near = [];             // last reveal result, reused between recomputes
  let nearest = null;        // the closest revealed item, for Enter
  let paused = false;        // a photo is open full-size
  let gal = null;            // the open gallery: { owner, list, i }

  let touchHold = false;     // a finger is on the screen: the touch throttle

  let restore = null;        // what to put back on the way out

  const cards = new Map();   // photo key -> element, pooled across frames
  const chips = new Map();   // item id -> element

  let root = null, sky = null, band = null, hazeEl = null, worldEl = null;
  let elAlt = null, elSpeed = null, elName = null, elHint = null, rushEl = null, elThr = null;
  let burnerEl = null, elBurn = null, sndBtn = null;
  let intro = null, viewer = null;

  const el = (id) => document.getElementById(id);
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const lerp = (a, b, t) => a + (b - a) * t;

  /* ---------- the engine ----------
   *
   * Synthesised, not sampled. A jet is noise shaped by resonance, which is
   * what a filter is, and building it from parts means every knob is live:
   * the body deepens and the hiss rises with speed, and the afterburner is a
   * separate roar that comes in under the throttle key. No file to fetch,
   * nothing to license, and the sound is the flight model's, not a loop
   * played over it. */

  const Engine = (() => {
    const KEY = 'dk.fly.sound';
    let ctx = null, master = null, n = null;
    let muted = false;
    try { muted = localStorage.getItem(KEY) === '0'; } catch (_) { /* private mode */ }

    function noise(seconds, brown) {
      const len = Math.floor(ctx.sampleRate * seconds);
      const buf = ctx.createBuffer(1, len, ctx.sampleRate);
      const d = buf.getChannelData(0);
      let b = 0;
      for (let i = 0; i < len; i++) {
        const w = Math.random() * 2 - 1;
        // Brown noise is white noise integrated with a leak, which is the
        // rumble under everything; white on its own is only the hiss.
        if (brown) { b = (b + 0.02 * w) / 1.02; d[i] = b * 3.5; } else d[i] = w;
      }
      return buf;
    }
    const loop = (buf) => {
      const s = ctx.createBufferSource();
      s.buffer = buf; s.loop = true; s.start();
      return s;
    };
    const gain = (v) => { const g = ctx.createGain(); g.gain.value = v; return g; };
    const filter = (type, f, q) => {
      const x = ctx.createBiquadFilter();
      x.type = type; x.frequency.value = f; if (q) x.Q.value = q;
      return x;
    };

    function build() {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return false;
      ctx = new AC();
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -18;
      comp.ratio.value = 6;
      master = gain(0);
      master.connect(comp);
      comp.connect(ctx.destination);

      const brown = noise(3, true), white = noise(2, false);

      // The body of the turbine: rumble through a low-pass whose cutoff climbs
      // with speed, so spooling up is heard as the sound opening.
      const coreLP = filter('lowpass', 160, 0.8);
      const coreG = gain(0.2);
      loop(brown).connect(coreLP); coreLP.connect(coreG); coreG.connect(master);

      // Air over the airframe: a band of white noise that is barely there at
      // a hover and most of the sound at top speed.
      const hissBP = filter('bandpass', 2600, 0.5);
      const hissG = gain(0.01);
      loop(white).connect(hissBP); hissBP.connect(hissG); hissG.connect(master);

      // The whine: a sawtooth and a sine an octave up, slightly off, so they
      // beat against each other the way real blades do. Pitch follows speed.
      const whineLP = filter('lowpass', 1500, 1.5);
      const whineG = gain(0.02);
      const o1 = ctx.createOscillator(); o1.type = 'sawtooth'; o1.frequency.value = 70;
      const o2 = ctx.createOscillator(); o2.type = 'sine'; o2.frequency.value = 141;
      o1.connect(whineLP); o2.connect(whineLP);
      whineLP.connect(whineG); whineG.connect(master);
      o1.start(); o2.start();

      // The afterburner: brown noise driven hard into a soft clipper, kept low,
      // with a slow tremolo so it crackles rather than hums. Silent until W.
      const shaper = ctx.createWaveShaper();
      const curve = new Float32Array(256);
      for (let i = 0; i < 256; i++) curve[i] = Math.tanh(3.2 * (i / 127.5 - 1));
      shaper.curve = curve;
      const burnLP = filter('lowpass', 240, 1.2);
      const burnG = gain(0);
      const trem = gain(1);
      const lfo = ctx.createOscillator(); lfo.type = 'sine'; lfo.frequency.value = 9.5;
      const lfoG = gain(0.35);
      lfo.connect(lfoG); lfoG.connect(trem.gain); lfo.start();
      loop(brown).connect(shaper); shaper.connect(burnLP); burnLP.connect(burnG);
      burnG.connect(trem); trem.connect(master);

      n = { coreLP, coreG, hissG, o1, o2, whineG, burnG };
      return true;
    }

    /** Start, or resume. Must be called from a user gesture the first time:
     *  browsers refuse to make a sound a page asked for on its own. */
    function start() {
      if (muted) return;
      if (!ctx && !build()) return;
      if (ctx.state === 'suspended') ctx.resume().catch(() => {});
      master.gain.cancelScheduledValues(ctx.currentTime);
      master.gain.setTargetAtTime(0.55, ctx.currentTime, 0.4);
    }

    function poke() {
      if (ctx && ctx.state === 'suspended' && !muted) ctx.resume().catch(() => {});
    }

    function stop() {
      if (!ctx) return;
      master.gain.cancelScheduledValues(ctx.currentTime);
      master.gain.setTargetAtTime(0, ctx.currentTime, 0.25);
      // Suspend once the fade is over, so a page with the mode closed is not
      // still running a synthesiser into silence.
      setTimeout(() => { if (ctx && master.gain.value < 0.01) ctx.suspend().catch(() => {}); }, 900);
    }

    /** Every frame. `s` is speed as a fraction of top speed, `b` the burner. */
    function set(s, b) {
      if (!ctx || muted || !n) return;
      const t = ctx.currentTime;
      s = clamp(s, 0, 1);
      n.coreLP.frequency.setTargetAtTime(150 + 950 * s, t, 0.1);
      n.coreG.gain.setTargetAtTime(0.2 + 0.4 * s, t, 0.1);
      n.hissG.gain.setTargetAtTime(0.01 + 0.2 * s * s, t, 0.1);
      const f = 70 + 330 * s;
      n.o1.frequency.setTargetAtTime(f, t, 0.15);
      n.o2.frequency.setTargetAtTime(f * 2.02, t, 0.15);
      n.whineG.gain.setTargetAtTime(0.015 + 0.05 * s, t, 0.1);
      n.burnG.gain.setTargetAtTime(0.65 * b, t, b ? 0.12 : 0.3);
    }

    function toggle() {
      muted = !muted;
      try { localStorage.setItem(KEY, muted ? '0' : '1'); } catch (_) { /* fine */ }
      if (muted) stop(); else start();
      return !muted;
    }

    return { start, stop, set, poke, toggle, isOn: () => !muted };
  })();

  /* ---------- geometry ---------- */

  /** Where you end up going `m` metres on `brg` from a point. Equirectangular,
   *  which over the few kilometres a flight covers is exact to centimetres. */
  function destination(lat, lng, brg, m) {
    const dN = m * Math.cos(brg * RAD);
    const dE = m * Math.sin(brg * RAD);
    return {
      lat: lat + dN / M_PER_DEG_LAT,
      lng: lng + dE / (M_PER_DEG_LAT * Math.cos(lat * RAD))
    };
  }

  const angleDiff = (a, b) => {
    let d = (a - b) % 360;
    if (d > 180) d -= 360;
    if (d < -180) d += 360;
    return d;
  };

  /** The zoom that puts the camera exactly `alt` metres above the ground.
   *
   *  MapLibre holds the camera `focal * height` pixels from the centre point,
   *  along the view axis; the vertical leg of that is `cos(pitch)` of it. So
   *  the metres-per-pixel we need is alt / (that leg), and the zoom is what
   *  gives that scale at this latitude. Inverting the projection like this,
   *  rather than picking zooms by eye, is what lets the altitude readout mean
   *  something and the photos size themselves in real metres. */
  function zoomFor(a, pitch, lat) {
    const h = map.getContainer().clientHeight || 800;
    const mpp = a / (focal * h * Math.cos(pitch * RAD));
    const worldPx = 40075016.686 * Math.cos(lat * RAD) / mpp;
    return clamp(Math.log2(worldPx / 512), 1, 22);
  }

  /** Screen y of the horizon, in map-container pixels. Points at infinity sit
   *  `focal / tan(pitch)` heights above the centre of the view. */
  function horizonY(pitch) {
    const h = map.getContainer().clientHeight || 800;
    return h * (0.5 - focal / Math.tan(pitch * RAD));
  }

  /** Screen pixels per world metre at a given place on the ground.
   *
   *  Measured rather than derived: project the point and a point 40 m to its
   *  side, and take the distance between them. That gets perspective, terrain
   *  and whatever else the transform is doing for free, and it is two matrix
   *  multiplies. The side-step is perpendicular to the heading so it is never
   *  foreshortened by the tilt. */
  function scaleAt(lngLat, screen) {
    const q = destination(lngLat[1], lngLat[0], bearing + 90, 40);
    const p2 = map.project([q.lng, q.lat]);
    return Math.hypot(p2.x - screen.x, p2.y - screen.y) / 40;
  }

  /** Nearest point of a polyline to (px, py), all in local metres. */
  function nearestOn(xy, px, py) {
    if (xy.length === 1) {
      return { d: Math.hypot(px - xy[0][0], py - xy[0][1]), x: xy[0][0], y: xy[0][1] };
    }
    let best = Infinity, bx = xy[0][0], by = xy[0][1];
    for (let i = 0; i < xy.length - 1; i++) {
      const ax = xy[i][0], ay = xy[i][1];
      const dx = xy[i + 1][0] - ax, dy = xy[i + 1][1] - ay;
      const len2 = dx * dx + dy * dy;
      let t = len2 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const qx = ax + t * dx, qy = ay + t * dy;
      const d2 = (px - qx) ** 2 + (py - qy) ** 2;
      if (d2 < best) { best = d2; bx = qx; by = qy; }
    }
    return { d: Math.sqrt(best), x: bx, y: by };
  }

  /* ---------- the frame ----------
   *
   * Rolling the picture is done in CSS, because MapLibre 4 has no roll axis,
   * and a rotated rectangle only keeps covering the screen if it is big
   * enough: for any angle at all, a square whose side is the screen's
   * diagonal. So on a desktop the map is grown to that square, hanging well
   * past every edge, and the window shows its middle.
   *
   * That would narrow the view - the window is now a small part of the
   * container, and MapLibre sets its camera distance from the container's
   * height - so the field of view is widened by exactly the amount that puts
   * the same picture back in the window. It is the same pinhole camera with a
   * larger sensor behind it; nothing in the visible frame changes, there is
   * just more of it painted off-screen for the roll to bring in. */

  function getFov() {
    if (typeof map.getVerticalFieldOfView === 'function') return map.getVerticalFieldOfView();
    return map.transform ? map.transform.fov : 36.87;
  }

  function setFov(deg) {
    if (typeof map.setVerticalFieldOfView === 'function') map.setVerticalFieldOfView(deg);
    else if (map.transform) map.transform.fov = deg;
  }

  function fitFrame() {
    const w = innerWidth, h = innerHeight;
    canRoll = matchMedia('(pointer: fine)').matches;
    if (canRoll) {
      const d = Math.ceil(Math.hypot(w, h));
      ox = Math.ceil((d - w) / 2);
      oy = Math.ceil((d - h) / 2);
    } else {
      ox = 0;
      oy = 0;
    }
    const rs = document.documentElement.style;
    rs.setProperty('--fly-ox', `${ox}px`);
    rs.setProperty('--fly-oy', `${oy}px`);
    // A retina screen would have the map fill four device pixels for every
    // css pixel of a canvas that is already twice the screen. Satellite tiles
    // at flying speed do not repay that, and the frame time does.
    if (map.setPixelRatio) map.setPixelRatio(Math.min(devicePixelRatio || 1, PIXEL_CAP));
    map.resize();

    const ch = map.getContainer().clientHeight || (h + 2 * oy);
    const dist = 0.5 * h / Math.tan((VFOV / 2) * RAD);
    // MapLibre caps the field of view at 60 degrees. An ultrawide screen asks
    // for more and gets a slightly narrower view instead of a broken one;
    // `focal` is taken from what was actually set, so the geometry stays true.
    const fov = clamp((2 * Math.atan((0.5 * ch) / dist)) / RAD, 10, 60);
    setFov(fov);
    focal = 0.5 / Math.tan((fov / 2) * RAD);
  }

  /* ---------- the world, precomputed once per flight ----------
   *
   * Everything the reveal test needs, in a flat local metric frame so that a
   * frame costs arithmetic and no trigonometry. Rebuilt on entry rather than
   * kept live: layers can be switched while flying only by leaving first. */

  let refLat = 0, refLng = 0, mPerLng = 0;

  const toLocal = (lat, lng) => [(lng - refLng) * mPerLng, (lat - refLat) * M_PER_DEG_LAT];
  const toLngLat = (x, y) => [refLng + x / mPerLng, refLat + y / M_PER_DEG_LAT];

  /** A point a given fraction along a path, by arc length. Photos of one trail
   *  are spread along it instead of stacked on its first vertex - a trail with
   *  four pictures should read as four places, which is what it is. */
  function alongPath(path, frac) {
    if (path.length === 1) return path[0];
    const segs = [];
    let total = 0;
    for (let i = 0; i < path.length - 1; i++) {
      const d = Math.hypot(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0]);
      segs.push(d);
      total += d;
    }
    let want = total * frac, acc = 0;
    for (let i = 0; i < segs.length; i++) {
      if (acc + segs[i] >= want) {
        const t = segs[i] ? (want - acc) / segs[i] : 0;
        return [path[i][0] + (path[i + 1][0] - path[i][0]) * t,
                path[i][1] + (path[i + 1][1] - path[i][1]) * t];
      }
      acc += segs[i];
    }
    return path[path.length - 1];
  }

  /** The pictures and videos of an item, in the order the app shows them. A
   *  video is a photo entry with `yt`; it floats as its thumbnail and plays
   *  when opened. */
  const mediaOf = (entry) =>
    (entry.item.photos || []).filter((p) => p && (p.yt || p.thumb || p.full));

  function buildWorld() {
    refLat = pos.lat;
    refLng = pos.lng;
    mPerLng = M_PER_DEG_LAT * Math.cos(refLat * RAD);

    world = [];
    shots = [];
    near = [];
    nearest = null;
    seen.clear();

    const items = [...Layers.visibleSegments(), ...Layers.visibleWaypoints()];
    const REACH = 30000;   // Houten and Curitiba are on the same map and are not here

    for (const it of items) {
      const path = it.path && it.path.length ? it.path
        : (it.lat != null && it.lng != null ? [[it.lat, it.lng]] : null);
      if (!path) continue;

      const xy = path.map(([lat, lng]) => toLocal(lat, lng));
      if (Math.hypot(xy[0][0], xy[0][1]) > REACH) continue;

      const entry = { id: it.id, name: it.name || '', item: it, xy, line: path.length > 1 };
      world.push(entry);

      const photos = mediaOf(entry).filter((p) => p.thumb || p.full);
      photos.forEach((p, i) => {
        // A waypoint has one coordinate and possibly several pictures, so they
        // are fanned onto a small ring rather than left in one pile.
        let at;
        if (path.length > 1) {
          at = alongPath(path, (i + 0.5) / photos.length);
        } else {
          const a = (i / Math.max(photos.length, 1)) * 360 + 40;
          const d = destination(path[0][0], path[0][1], a, photos.length > 1 ? 26 : 0);
          at = [d.lat, d.lng];
        }
        const [x, y] = toLocal(at[0], at[1]);
        shots.push({
          key: `${it.id}:${i}`,
          owner: entry,
          name: it.name || '',
          cap: p.cap || '',
          src: p.thumb || p.full,
          full: p.full || p.thumb,
          yt: p.yt || null,
          lngLat: [at[1], at[0]],
          x, y
        });
      });
    }
  }

  /* ---------- reveal ----------
   *
   * Measured from a point ahead of the flyer rather than from underneath: at
   * this tilt, what is directly below is at the very bottom of the screen and
   * mostly out of it, and revealing things you cannot see is the same as not
   * revealing them. */

  function focusPoint(pitch) {
    const ahead = alt * Math.tan(pitch * RAD) * 0.55;
    return destination(pos.lat, pos.lng, bearing, ahead);
  }

  const EMPTY = { type: 'FeatureCollection', features: [] };

  function recompute(pitch, now) {
    const f = focusPoint(pitch);
    const [fx, fy] = toLocal(f.lat, f.lng);
    const reach = clamp(alt * REVEAL_K, REVEAL_MIN, REVEAL_MAX);

    near = [];
    for (const e of world) {
      const hit = nearestOn(e.xy, fx, fy);
      if (hit.d > reach) { seen.delete(e.id); continue; }
      if (!seen.has(e.id)) seen.set(e.id, now);
      // Eased so a trail arrives as a glow that swells rather than a line that
      // switches on, and a pulse for the first second it is in range: finding
      // something should be an event.
      const t = 1 - hit.d / reach;
      const pulse = Math.exp(-(now - seen.get(e.id)) / 0.85) * 0.75;
      e.g = clamp(t * t * (3 - 2 * t) + pulse, 0, 1);
      e.d = hit.d;
      e.at = toLngLat(hit.x, hit.y);
      near.push(e);
    }
    near.sort((a, b) => a.d - b.d);
    nearest = near[0] || null;

    const src = map.getSource('fly-trails');
    if (src) {
      src.setData({
        type: 'FeatureCollection',
        features: near.filter((e) => e.line).slice(0, MAX_LINES).map((e) => ({
          type: 'Feature',
          properties: { g: e.g, h: e.g * 0.7 },
          geometry: {
            type: 'LineString',
            coordinates: e.item.path.map(([lat, lng]) => [lng, lat])
          }
        }))
      });
    }

    paintChips();
    return { fx, fy };
  }

  /* ---------- floating photos ----------
   *
   * Sized in world metres and projected, so approaching one really does make
   * it bigger - the growth is perspective, not a distance curve. The card
   * hangs above its point on a tether, because a picture lying flat on a
   * trail at this tilt is a smear and a picture with nothing under it belongs
   * nowhere. */

  function cardFor(shot) {
    let node = cards.get(shot.key);
    if (node) return node;
    node = document.createElement('div');
    node.className = 'fly-card' + (shot.yt ? ' video' : '');
    // The halo is the click target, wider than the card by a finger's width:
    // the card is moving while you aim at it, and a target that has to be hit
    // exactly is not a target you can hit from a moving aircraft.
    node.innerHTML =
      `<i class="fly-card-hit"></i>` +
      `<div class="fly-card-img"><img alt="" decoding="async" referrerpolicy="no-referrer"></div>` +
      `<div class="fly-card-cap"></div>`;
    const img = node.querySelector('img');
    // A picture that will not load is a black rectangle hanging over a trail,
    // which reads as a fault in the trail rather than in the file. Mark it and
    // it stops being a place you can fly to.
    img.addEventListener('error', () => { shot.dead = true; node.hidden = true; });
    img.src = shot.src;
    node.querySelector('.fly-card-cap').textContent = shot.name;
    node.addEventListener('click', (e) => { e.stopPropagation(); openShot(shot); });
    worldEl.appendChild(node);
    cards.set(shot.key, node);
    return node;
  }

  function paintCards(pitch, fx, fy) {
    const reach = clamp(alt * PHOTO_K, PHOTO_MIN, PHOTO_MAX);
    const vh = map.getContainer().clientHeight || 800;
    const sinP = Math.sin(pitch * RAD);

    // World size of a card, and how high above the ground it floats. Both
    // follow altitude so that a card is a similar size on screen whether you
    // are skimming or surveying - what changes with distance, and only that,
    // is how much bigger it gets as you approach it.
    const wMetres = clamp(alt * 0.20, 22, 90);
    const hMetres = clamp(alt * 0.18, 20, 90);

    // The window the viewer can actually see, in the map's own pixels. The map
    // hangs past the screen on every side, so its centre and the screen's
    // centre coincide but its edges are outside. Once the picture is rolled
    // the window is no longer axis-aligned in these pixels, and the whole
    // container is used instead: a few cards painted where nobody sees them
    // are cheaper than a card missing from a corner that has just rolled in.
    const cw = map.getContainer().clientWidth;
    const rolled = Math.abs(bank) > 6;
    const x0 = rolled ? -40 : ox - 40;
    const x1 = rolled ? cw + 40 : ox + innerWidth + 40;
    const y1 = rolled ? vh + 40 : oy + innerHeight + 40;

    const [mx, my] = toLocal(pos.lat, pos.lng);
    const live = [];
    for (const s of shots) {
      if (s.dead) continue;
      const d = Math.hypot(s.x - fx, s.y - fy);
      if (d > reach) continue;
      // Behind you is not a view. The projection folds points behind the
      // camera plane back onto the screen, so this cull is correctness and
      // not only economy.
      const brgTo = Math.atan2(s.x - mx, s.y - my) / RAD;
      if (Math.abs(angleDiff(brgTo, bearing)) > 88) continue;
      s.d = d;
      live.push(s);
    }
    live.sort((a, b) => a.d - b.d);

    const skyY = horizonY(pitch);
    const keep = new Set();
    const placed = [];

    for (const s of live) {
      if (placed.length >= MAX_CARDS) break;
      const p = map.project(s.lngLat);
      if (!isFinite(p.x) || !isFinite(p.y)) continue;
      // Right on the horizon a card is a smudge at the very top of the screen,
      // and it is the tether reaching up out of the frame that you notice, not
      // the picture. Those distances are the reveal radius doing its job; they
      // do not also need a photograph.
      if (p.y < skyY + 26) continue;

      const ppm = scaleAt(s.lngLat, p);
      if (!isFinite(ppm) || ppm <= 0) continue;

      const w = clamp(wMetres * ppm, CARD_MIN_PX, innerHeight * CARD_MAX_VH);
      const tether = clamp(hMetres * ppm * sinP, 6, innerHeight * 0.22);
      const cardH = w * 0.75 + 26;

      // Off the screen is off the screen. The heading test above only rejects
      // what is behind the camera; a photo a little to the side and close by
      // passes it and still lands two thousand pixels below the bottom edge,
      // because at this tilt the ground under you is not in the picture. Those
      // were counting against the ten slots and showing nothing.
      if (p.x + w / 2 < x0 || p.x - w / 2 > x1) continue;
      if (p.y - tether - cardH > y1) continue;

      // Nearest wins the spot. Cards are placed closest-first, so a photo that
      // would land on top of one already placed is the further of the two and
      // is the one to drop - otherwise a trail with eight pictures becomes one
      // illegible pile and hides the trail beside it.
      let clash = false;
      for (const q of placed) {
        if (Math.abs(p.x - q.x) < (w + q.w) * 0.5 &&
            Math.abs(p.y - q.y) < (w + q.w) * 0.4) { clash = true; break; }
      }
      if (clash) continue;
      placed.push({ x: p.x, y: p.y, w });
      // Fades in over the outer fifth of the range, so cards arrive rather
      // than pop, and never fully vanishes while in range.
      const fade = clamp((1 - s.d / reach) * 5, 0, 1);

      const node = cardFor(s);
      node.style.transform =
        `translate3d(${p.x.toFixed(1)}px, ${(p.y - tether).toFixed(1)}px, 0) translate(-50%, -100%)`;
      node.style.width = `${w}px`;
      node.style.setProperty('--tether', `${tether}px`);
      node.style.opacity = fade;
      node.style.zIndex = String(4000 - Math.round(s.d));
      node.classList.toggle('small', w < 88);
      node.hidden = false;
      keep.add(s.key);
    }

    for (const [key, node] of cards) {
      if (!keep.has(key)) node.hidden = true;
    }
  }

  function paintChips() {
    const keep = new Set();
    // Five names fit across a laptop and pile on top of each other across a
    // phone, where they are also the widest thing on the screen.
    const cap = innerWidth < 560 ? 2 : MAX_CHIPS;
    const pick = near.filter((e) => e.g > 0.42 && e.name).slice(0, cap);
    const vh = map.getContainer().clientHeight || 800;
    // The bottom strip belongs to the instruments, and the nearest trail's name
    // is already printed there. A chip that lands under it prints the same
    // words twice, half of each behind the other.
    const floor = (vh - innerHeight) / 2 + innerHeight - 96;

    for (const e of pick) {
      const p = map.project([e.at[0], e.at[1]]);
      if (!isFinite(p.x) || !isFinite(p.y)) continue;
      let node = chips.get(e.id);
      if (!node) {
        node = document.createElement('div');
        node.className = 'fly-chip';
        worldEl.appendChild(node);
        chips.set(e.id, node);
      }
      node.textContent = e.name;
      node.style.transform =
        `translate3d(${p.x.toFixed(1)}px, ${p.y.toFixed(1)}px, 0) translate(-50%, -175%)`;
      node.style.fontSize =
        `${clamp(CHIP_MIN_PX + (e.g - 0.42) * 34, CHIP_MIN_PX, CHIP_MAX_PX)}px`;
      node.style.opacity = clamp((e.g - 0.42) * 3.4, 0, 1);
      node.style.zIndex = String(3000 - Math.round(e.d));
      node.hidden = p.y < -80 || p.y > floor;
      keep.add(e.id);
    }
    for (const [id, node] of chips) if (!keep.has(id)) node.hidden = true;
  }

  /* ---------- the flight ---------- */

  /** The stick's response: nothing inside the dead zone, then a curve that is
   *  gentle near the middle and firm at the edge, so small corrections are
   *  possible and a hard turn is still there when it is wanted. */
  function stick(v) {
    const a = Math.abs(v);
    if (a <= DEAD) return 0;
    const s = (a - DEAD) / (1 - DEAD);
    return Math.sign(v) * Math.pow(s, 1.6);
  }

  function step(now) {
    raf = requestAnimationFrame(step);
    const dt = clamp((now - last) / 1000, 0, 0.06);
    last = now;
    // During the take-off ease the camera belongs to MapLibre, but the sky
    // still has to follow the horizon it is climbing towards.
    if (!flying) { paintHud(map.getPitch()); return; }

    const t = clamp((alt - ALT_MIN) / (ALT_MAX - ALT_MIN), 0, 1);
    const pitch = clamp(lerp(PITCH_LOW, PITCH_HIGH, Math.sqrt(t)) + pitchTrim,
                        PITCH_MIN, PITCH_MAX);

    const cruise = clamp(alt * SPEED_K, SPEED_MIN, SPEED_MAX);
    const top = cruise * BOOST;
    let burnWant = 0;

    // A photo open full-size holds everything where it is. The speed you had
    // is the speed you get back, because closing a picture is not landing.
    if (!paused) {
      // Throttle. The button held opens it, the button released lets it wind
      // down, and the speed follows the throttle rather than the button, so a
      // press is a push rather than a switch. Rise is quicker than fall on
      // purpose: a rhythm of short presses holds a speed without effort, and
      // letting go altogether is a slow coast to a stop, not a brake. Top
      // speed follows altitude, so a dive at full throttle does not arrive at
      // the rooftops at survey speed.
      const open = hold || touchHold;
      throttle = open ? Math.min(1, throttle + dt / THR_UP)
                      : Math.max(0, throttle - dt / THR_DOWN);
      speed += (throttle * top - speed) * clamp(ACCEL * dt, 0, 1);
      // The burner is the top of the throttle, past cruise. Below it, with
      // the button down, the engine is heard spooling, so the press is
      // answered the moment it lands.
      burnWant = throttle * BOOST > 1.02 ? 1 : open ? 0.45 : 0;

      // The stick. The mouse asks nothing while it rests on a photo, on a
      // button, on the help card, or outside the window: those are the moments
      // it is being used as a mouse.
      // A card that slid out of range while the cursor rested on it would
      // otherwise hold the stick until the mouse next moved.
      if (hover.card && hover.card.hidden) hover.card = null;
      const held = !!hover.card && now - hover.since > HOVER_MS;   // the cursor is on a photo
      const steer = canRoll && mouse.in && armed && intro.hidden && !hover.hud && !held;
      const sx = steer ? stick(mouse.nx) : 0;
      const keyTurn = (keys.has('ArrowRight') ? 1 : 0) - (keys.has('ArrowLeft') ? 1 : 0);
      const turnIn = clamp(sx + keyTurn, -1, 1);

      // Roll. A and D roll at a fixed rate for as long as they are down, past
      // the vertical and over the top if you like. Released, the wings level
      // themselves - the short way round, so a roll let go past inverted
      // finishes rather than unwinds - onto the bank a plain turn would show.
      const rollIn = canRoll ? (keys.has('KeyD') ? 1 : 0) - (keys.has('KeyA') ? 1 : 0) : 0;
      if (rollIn) {
        bankRate += (rollIn * ROLL_RATE - bankRate) * clamp(ROLL_ACCEL * dt, 0, 1);
        bank += bankRate * dt;
      } else {
        bankRate = 0;
        const want = canRoll ? turnIn * BANK_MAX : 0;
        bank += angleDiff(want, bank) * clamp(LEVEL_K * dt, 0, 1);
      }
      bank = angleDiff(bank, 0);

      // Turn: what the stick asks, plus what the bank gives. A bank turns the
      // nose the way lift does, so a rolled-in turn is tighter than a flat one
      // and an inverted aircraft goes straight - which is also what makes a
      // full roll come out pointing where it went in.
      const wantYaw = turnIn * YAW_RATE + Math.sin(bank * RAD) * BANK_TURN;
      // The turn dies quickly under a held photo, so it stops sliding away
      // from the cursor that is trying to click it.
      yaw += (wantYaw - yaw) * clamp((held ? 12 : YAW_ACCEL) * dt, 0, 1);
      bearing = (bearing + yaw * dt + 360) % 360;

      // Elevator. W pushes the nose down and dives, S pulls it up and climbs,
      // the way a stick pushed forward does; the arrows climb and sink without
      // moving the view. Climb rate is a fraction of the height you are at, so
      // the whole range from rooftop to survey takes a handful of seconds
      // either way.
      const elev = (keys.has('KeyS') ? 1 : 0) - (keys.has('KeyW') ? 1 : 0);
      const trimWant = elev > 0 ? TRIM_UP : elev < 0 ? -TRIM_DOWN : 0;
      pitchTrim += (trimWant - pitchTrim) * clamp(4 * dt, 0, 1);
      const climb = (keys.has('ArrowUp') ? 1 : 0) - (keys.has('ArrowDown') ? 1 : 0) + elev;
      if (climb) alt = clamp(alt * (1 + climb * CLIMB_K * dt) + climb * 6 * dt,
                             ALT_MIN, ALT_MAX);
      // Banked over, the wings hold less. Gentle enough that a mouse turn's
      // bank costs nothing you would notice and a knife-edge costs a little.
      const lift = Math.cos(bank * RAD);
      if (lift < 0.9) alt = clamp(alt * (1 - (1 - lift) * SINK_K * dt), ALT_MIN, ALT_MAX);

      if (speed) pos = destination(pos.lat, pos.lng, bearing, speed * dt);
    }

    burn += (burnWant - burn) * clamp((burnWant > burn ? 6 : 3) * dt, 0, 1);

    // Bank the picture. The map, the sky and the floating cards are rotated
    // together in CSS - they have to move as one or the photos slide off
    // their trails. Right wing down is a counter-clockwise picture.
    if (canRoll) {
      const tf = `rotate(${(-bank).toFixed(2)}deg)`;
      map.getContainer().style.transform = tf;
      worldEl.style.transform = tf;
      sky.style.transform = tf;
    }

    const ahead = destination(pos.lat, pos.lng, bearing, alt * Math.tan(pitch * RAD));
    map.jumpTo({
      center: [ahead.lng, ahead.lat],
      zoom: zoomFor(alt, pitch, ahead.lat),
      bearing,
      pitch
    });

    // The reveal test is the expensive half and does not need every frame; the
    // positions of what it revealed do, or the cards swim behind the camera.
    tick -= dt;
    let fx, fy;
    if (tick <= 0) {
      tick = 0.11;
      ({ fx, fy } = recompute(pitch, now / 1000));
    } else {
      const f = focusPoint(pitch);
      [fx, fy] = toLocal(f.lat, f.lng);
      paintChips();   // on last frame's reveal: what was found has not changed,
    }                 // but where it is on the screen has
    paintCards(pitch, fx, fy);
    paintHud(pitch);

    // Every other frame is plenty for an automation timeline; the ramps are
    // smoothed on the audio thread anyway.
    if ((frame++ & 1) === 0) Engine.set(paused ? 0.12 : Math.abs(speed) / top, paused ? 0 : burn);
  }

  function paintHud(pitch) {
    // The sky is a band that ends exactly at the horizon, so the pale end of
    // its gradient always meets the ground rather than landing wherever a
    // fixed gradient happened to put it. It sits behind the map, so the seam
    // itself is never seen; the haze in front of the map covers where the far
    // tiles stop.
    //
    // Both are moved with a transform and never resized or repositioned: a
    // painted layer the size of the map that changes every frame has to be
    // rasterised again every frame, and when Chrome falls behind on that it
    // shows the unpainted parts as nothing at all. That was the flicker.
    const y = horizonY(pitch);
    hazeEl.style.transform = `translate3d(0, ${y.toFixed(1)}px, 0)`;
    band.style.transform = `translate3d(0, ${(y + 6 - BAND_H).toFixed(1)}px, 0)`;

    elAlt.textContent = `${Math.round(alt)} מ׳`;
    elSpeed.textContent = `${Math.round(Math.abs(speed) * 3.6)} קמ״ש`;
    elThr.style.width = `${(throttle * 100).toFixed(0)}%`;

    const top = clamp(alt * SPEED_K, SPEED_MIN, SPEED_MAX) * BOOST;
    rushEl.style.opacity = clamp((Math.abs(speed) / top - 0.34) * 0.62, 0, 0.34);
    burnerEl.style.opacity = burn * 0.85;
    document.body.classList.toggle('fly-burn', burn > 0.5);
    elBurn.hidden = burn < 0.5;

    if (nearest && nearest.name) {
      elName.textContent = nearest.name;
      elName.hidden = false;
      elHint.hidden = false;
      elHint.classList.toggle('open', !mediaOf(nearest).length);
    } else {
      elName.hidden = true;
      elHint.hidden = true;
    }
  }

  /* ---------- looking at one photo, or all of them ----------
   *
   * The viewer is a gallery of the item the card belongs to, not the one
   * picture: a trail with six photos and a video is one place, and having
   * flown to it you should be able to see all of it without landing. Videos
   * play in place. The flight is held, not stopped, while it is open. */

  function openShot(shot) {
    const list = mediaOf(shot.owner);
    let i = list.findIndex((p) => (shot.yt ? p.yt === shot.yt : (p.full || p.thumb) === shot.full));
    openGallery(shot.owner, list, i < 0 ? 0 : i);
  }

  function openGallery(owner, list, i) {
    if (!list.length) return;
    gal = { owner, list, i };
    paused = true;
    // The instruments go while a picture is being looked at. They report a
    // flight that is standing still, and their close button sits in the same
    // corner as the viewer's own - two crosses on top of each other, neither
    // of them obviously the one that closes what is in front of you.
    document.body.classList.add('fly-paused');
    viewer.dataset.item = owner.id;
    viewer.hidden = false;
    paintGallery();
  }

  /** Same lesson the app's lightbox learned: `src = ''` on an iframe resolves
   *  to this page and loads a second copy of the app inside the frame.
   *  about:blank is a real navigation away from YouTube, which is what
   *  actually stops the sound. */
  function blankVideo(vid) {
    if (vid.getAttribute('src')) vid.src = 'about:blank';
    vid.removeAttribute('src');
    vid.hidden = true;
  }

  function paintGallery() {
    const p = gal.list[gal.i];
    const img = viewer.querySelector('.fly-view-img');
    const vid = viewer.querySelector('.fly-view-video');
    if (p.yt) {
      img.hidden = true;
      img.removeAttribute('src');
      vid.hidden = false;
      // autoplay, because getting here took a deliberate press; nocookie so
      // that looking does not set a tracking cookie for somebody who never
      // pressed play.
      vid.src = 'https://www.youtube-nocookie.com/embed/'
        + encodeURIComponent(p.yt) + '?autoplay=1&rel=0';
    } else {
      blankVideo(vid);
      img.hidden = false;
      img.src = p.full || p.thumb;
    }
    viewer.querySelector('.fly-view-name').textContent = gal.owner.name;
    const cap = viewer.querySelector('.fly-view-cap');
    cap.textContent = p.cap || '';
    cap.hidden = !p.cap;
    const many = gal.list.length > 1;
    viewer.querySelector('.fly-view-count').textContent =
      many ? `${gal.i + 1} / ${gal.list.length}` : '';
    viewer.querySelector('.fly-view-prev').hidden = !many;
    viewer.querySelector('.fly-view-next').hidden = !many;
  }

  function stepGallery(d) {
    if (!gal) return;
    gal.i = (gal.i + d + gal.list.length) % gal.list.length;
    paintGallery();
  }

  function closeShot() {
    document.body.classList.remove('fly-paused');
    viewer.hidden = true;
    viewer.querySelector('.fly-view-img').removeAttribute('src');
    blankVideo(viewer.querySelector('.fly-view-video'));
    gal = null;
    paused = false;
    disarm();   // the cursor is on the close button, not on the stick
  }

  /** Enter: the pictures of what you are over, without landing. */
  function openNearest() {
    if (!nearest) return;
    const list = mediaOf(nearest);
    if (list.length) openGallery(nearest, list, 0);
    else leaveTo(nearest.id);
  }

  /** Land on an item: leave the mode and open its page. */
  function leaveTo(id) {
    if (!id) return;
    if (viewer && !viewer.hidden) closeShot();
    exit({ keepPlace: true });
    setTimeout(() => select(id), 320);
  }

  /* ---------- map layers ---------- */

  function addLayers() {
    if (!map.getSource('fly-trails')) {
      map.addSource('fly-trails', { type: 'geojson', data: EMPTY });
    }
    if (!map.getLayer('fly-trail-glow')) {
      map.addLayer({
        id: 'fly-trail-glow',
        type: 'line',
        source: 'fly-trails',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': '#ffd166',
          'line-width': ['interpolate', ['linear'], ['zoom'], 13, 9, 18, 36],
          'line-blur': ['interpolate', ['linear'], ['zoom'], 13, 6, 18, 20],
          'line-opacity': ['get', 'h']
        }
      });
    }
    if (!map.getLayer('fly-trail-core')) {
      map.addLayer({
        id: 'fly-trail-core',
        type: 'line',
        source: 'fly-trails',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': '#fffaea',
          'line-width': ['interpolate', ['linear'], ['zoom'], 13, 2.4, 18, 8],
          'line-opacity': ['get', 'g']
        }
      });
    }
  }

  function removeLayers() {
    for (const id of ['fly-trail-core', 'fly-trail-glow']) {
      if (map.getLayer(id)) map.removeLayer(id);
    }
    if (map.getSource('fly-trails')) map.removeSource('fly-trails');
  }

  /** Hide everything the app normally draws, and remember what was hidden so
   *  the way out puts back exactly what was there. Nothing is drawn until you
   *  find it - that is the whole mechanic, and a map with all 62 trails
   *  already on it has nothing left to discover. */
  function hideOverlays() {
    const was = [];
    for (const layer of map.getStyle().layers) {
      if (layer.id.startsWith('fly-') || layer.source === 'sat') continue;
      const vis = map.getLayoutProperty(layer.id, 'visibility');
      was.push([layer.id, vis]);
      map.setLayoutProperty(layer.id, 'visibility', 'none');
    }
    return was;
  }

  function showOverlays(was) {
    for (const [id, vis] of was) {
      if (!map.getLayer(id)) continue;
      map.setLayoutProperty(id, 'visibility', vis === undefined ? 'visible' : vis);
    }
  }

  /* ---------- input ---------- */

  /* Keyed on `code` and never on `key`: the layout here is Hebrew half the
   * time, and on a Hebrew layout `key` for the W position is 'ט'. `code` is
   * the physical key and is the same either way. */

  function onKeyDown(e) {
    if (!on) return;
    if (e.code === 'Escape') {
      e.preventDefault();
      if (!viewer.hidden) closeShot(); else exit();
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (!viewer.hidden) {
      // The app's convention, RTL: left is onward.
      if (e.code === 'ArrowLeft') { e.preventDefault(); stepGallery(1); }
      else if (e.code === 'ArrowRight') { e.preventDefault(); stepGallery(-1); }
      return;
    }
    if (e.code === 'Enter') {
      e.preventDefault();
      if (e.shiftKey) leaveTo(nearest && nearest.id); else openNearest();
      return;
    }
    if (e.code === 'KeyM') { e.preventDefault(); setSound(Engine.toggle()); return; }
    if (e.code === 'KeyH' || e.code === 'Slash') { e.preventDefault(); toggleIntro(); return; }
    if (/^(Key[WASD]|Arrow(Up|Down|Left|Right))$/.test(e.code)) {
      e.preventDefault();
      keys.add(e.code);
      dismissIntro();
      Engine.poke();   // a key is a gesture, for a context the browser left suspended
    }
  }

  function onKeyUp(e) {
    keys.delete(e.code);
  }

  /* The mouse is the stick: where it sits is where the nose goes, no button
   * held. It is read on every move and turned into a position relative to the
   * middle of the screen; what it does with that position is decided in
   * `step`, which also knows when the cursor is busy being a cursor. */

  function onPointerMove(e) {
    if (!on || e.pointerType === 'touch') return;
    const hw = innerWidth / 2, hh = innerHeight / 2;
    mouse.nx = clamp((e.clientX - hw) / hw, -1, 1);
    mouse.ny = clamp((e.clientY - hh) / hh, -1, 1);
    mouse.in = true;
    if (!armed && Math.abs(mouse.nx) < DEAD * 1.6 && Math.abs(mouse.ny) < DEAD * 1.6) {
      armed = true;
      document.body.classList.remove('fly-unarmed');
    }
    const t = e.target;
    hover.hud = !!(t && t.closest && t.closest('.fly-hud button, .fly-view, .fly-intro'));
    const card = t && t.closest ? t.closest('.fly-card') : null;
    if (card !== hover.card) {
      hover.card = card;
      hover.since = performance.now();
    }
  }

  // Out of the window is hands off the stick, and so is the window losing
  // focus; a cursor parked at the edge of the screen while you read something
  // else would otherwise fly you in circles.
  function onPointerLeave() {
    mouse.in = false;
    hover.card = null;
    hover.hud = false;
    hold = false;
    disarm();
  }

  /* The button is the throttle. Only the main button, and only over open
   * ground: a press on a photo is a click, a press on a button is a button.
   * The release is taken from anywhere, because a finger that slid off the
   * window is still a finger that let go. */
  function onPointerDown(e) {
    if (!on || e.pointerType === 'touch' || e.button !== 0) return;
    const t = e.target;
    if (t && t.closest && t.closest('.fly-card, .fly-hud, .fly-view, .fly-intro')) return;
    hold = true;
    Engine.poke();
  }

  function onPointerUp() { hold = false; }

  function disarm() {
    if (!canRoll) return;
    armed = false;
    document.body.classList.add('fly-unarmed');
  }

  function onBlur() {
    keys.clear();
    touchHold = false;
    onPointerLeave();
  }

  function onClick(e) {
    const t = e.target;
    if (t && t.closest && t.closest('.fly-card, .fly-hud, .fly-view, .fly-intro')) return;
    dismissIntro();
    Engine.poke();
  }

  /* Touch: no keyboard to fly with, so a drag steers and holding the screen
   * is the throttle. Deliberately small - this mode is a desktop pleasure and
   * a phone should get something that works rather than a second control
   * scheme to learn. */
  let touch = null;

  function onTouchStart(e) {
    const tgt = e.target;
    if (tgt && tgt.closest && tgt.closest('.fly-card, .fly-hud, .fly-view')) return;
    const t = e.touches[0];
    touch = { x: t.clientX, y: t.clientY };
    touchHold = true;
    dismissIntro();
    Engine.poke();
  }

  function onTouchMove(e) {
    if (!touch) return;
    const t = e.touches[0];
    const dx = t.clientX - touch.x, dy = t.clientY - touch.y;
    touch.x = t.clientX; touch.y = t.clientY;
    bearing = (bearing - dx * 0.28 + 360) % 360;
    if (Math.abs(dy) > 1) alt = clamp(alt * (1 + dy * 0.004), ALT_MIN, ALT_MAX);
  }

  function onTouchEnd() { touch = null; touchHold = false; }

  function onVisibility() {
    if (!on) return;
    if (document.hidden) { Engine.stop(); keys.clear(); } else Engine.start();
  }

  // A right-click over the map is a slip of the hand, and a context menu
  // over a flight is the browser stepping in front of it.
  function onContextMenu(e) {
    if (on && !e.target.closest('.fly-view')) e.preventDefault();
  }

  function setSound(isOn) {
    if (!sndBtn) return;
    sndBtn.classList.toggle('off', !isOn);
    sndBtn.setAttribute('aria-label', isOn ? 'השתקת המנוע' : 'הפעלת הצליל');
    sndBtn.title = isOn ? 'השתקה (M)' : 'צליל (M)';
  }

  /* ---------- intro ---------- */

  let introTimer = null;

  function dismissIntro() {
    if (!intro || intro.hidden) return;
    intro.hidden = true;
    clearTimeout(introTimer);
    disarm();
  }

  function toggleIntro() {
    intro.hidden = !intro.hidden;
    clearTimeout(introTimer);
    if (intro.hidden) disarm();
  }

  /* ---------- entering and leaving ---------- */

  function buildDom() {
    if (root) return;
    root = document.createElement('div');
    root.className = 'fly-root';
    root.innerHTML = `
      <div class="fly-sky" id="fly-sky"><div class="fly-band" id="fly-band"></div></div>
      <div class="fly-world" id="fly-world"><div class="fly-haze" id="fly-haze"></div></div>
      <div class="fly-rush" id="fly-rush"></div>
      <div class="fly-burner" id="fly-burner"></div>
      <div class="fly-vig"></div>
      <div class="fly-hud">
        <div class="fly-reticle" aria-hidden="true"></div>
        <div class="fly-readout">
          <span class="fly-gauge"><b id="fly-alt">—</b><i>גובה</i></span>
          <span class="fly-gauge"><b id="fly-speed">—</b><i>מהירות</i></span>
          <span class="fly-gauge burn" id="fly-burn" hidden><b>מבער</b><i>אחורי</i></span>
        </div>
        <div class="fly-thr" aria-hidden="true"><i id="fly-thr-fill"></i></div>
        <button class="fly-x" id="fly-x" aria-label="יציאה ממצב תעופה">&times;</button>
        <button class="fly-help" id="fly-help" aria-label="מקשים">?</button>
        <button class="fly-snd" id="fly-snd" aria-label="השתקת המנוע">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path class="snd-body" d="M4 9v6h4l5 4V5L8 9H4z"/>
            <path class="snd-wave" d="M16 8.5a4.5 4.5 0 010 7M18.5 5.5a8.5 8.5 0 010 13" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
            <path class="snd-off" d="M16.5 9.5l5 5m0-5l-5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </button>
        <div class="fly-found">
          <strong id="fly-name" hidden></strong>
          <span id="fly-hint" class="fly-hint" hidden>
            <span class="hint-media"><kbd>Enter</kbd> <span>לתמונות</span></span>
            <span class="hint-open"><kbd>⇧ Enter</kbd> <span>לפתיחה</span></span>
          </span>
        </div>
        <p class="fly-credit">Esri · Maxar · Earthstar Geographics · גובה: Mapzen / AWS</p>
      </div>
      <div class="fly-intro" id="fly-intro">
        <div class="fly-intro-card">
          <h2>מצב תעופה</h2>
          <p>אתה מרחף מעל פרדס חנה־כרכור. דרכי הקיצור נדלקות כשמתקרבים אליהן,
             והתמונות שלהן תלויות באוויר מעל השביל. טסים אל תמונה, ולוחצים עליה.</p>
          <ul class="fly-keys">
            <li><kbd class="wide">עכבר</kbd><span>ההגה: ימינה ושמאלה פונים</span></li>
            <li><kbd class="wide">לחיצה</kbd><span>המצערת: כל עוד הכפתור לחוץ המנוע מגביר, וכשמשחררים הוא דועך</span></li>
            <li><kbd>W</kbd><kbd>S</kbd><span>אף למטה, אף למעלה</span></li>
            <li><kbd>A</kbd><kbd>D</kbd><span>גלגול. להחזיק לגלגול שלם</span></li>
            <li><kbd>↑</kbd><kbd>↓</kbd><span>גובה</span></li>
            <li><kbd>Enter</kbd><span>התמונות של השביל הקרוב</span></li>
            <li><kbd>M</kbd><span>המנוע</span></li>
            <li><kbd>Esc</kbd><span>יציאה</span></li>
          </ul>
          <p class="fly-intro-foot">כשהסמן נח על תמונה ההגה משתחרר, כדי שאפשר יהיה ללחוץ עליה.</p>
          <p class="fly-intro-touch">אצבע על המסך היא המצערת: מחזיקים וטסים, משחררים ונעצרים.
             גרירה לצדדים פונה, למעלה ולמטה משנה גובה. נגיעה בתמונה פותחת אותה.</p>
        </div>
      </div>
      <div class="fly-view" id="fly-view" hidden>
        <button class="fly-view-x" aria-label="סגירה">&times;</button>
        <button class="fly-view-nav fly-view-prev" aria-label="הקודמת">&rsaquo;</button>
        <button class="fly-view-nav fly-view-next" aria-label="הבאה">&lsaquo;</button>
        <div class="fly-view-stage">
          <img class="fly-view-img" alt="" referrerpolicy="no-referrer">
          <iframe class="fly-view-video" hidden allow="autoplay; encrypted-media; picture-in-picture"
                  allowfullscreen referrerpolicy="strict-origin-when-cross-origin" title="סרטון"></iframe>
        </div>
        <div class="fly-view-foot">
          <strong class="fly-view-name"></strong>
          <span class="fly-view-cap"></span>
          <span class="fly-view-count"></span>
          <button class="fly-view-open">פתח את השביל</button>
        </div>
      </div>`;
    document.body.appendChild(root);

    sky = el('fly-sky');
    band = el('fly-band');
    worldEl = el('fly-world');
    hazeEl = el('fly-haze');
    rushEl = el('fly-rush');
    burnerEl = el('fly-burner');
    elAlt = el('fly-alt');
    elSpeed = el('fly-speed');
    elBurn = el('fly-burn');
    elThr = el('fly-thr-fill');
    elName = el('fly-name');
    elHint = el('fly-hint');
    sndBtn = el('fly-snd');
    intro = el('fly-intro');
    viewer = el('fly-view');

    el('fly-x').addEventListener('click', () => exit());
    el('fly-help').addEventListener('click', toggleIntro);
    sndBtn.addEventListener('click', () => setSound(Engine.toggle()));
    intro.addEventListener('click', dismissIntro);
    viewer.querySelector('.fly-view-x').addEventListener('click', closeShot);
    viewer.querySelector('.fly-view-prev').addEventListener('click', () => stepGallery(-1));
    viewer.querySelector('.fly-view-next').addEventListener('click', () => stepGallery(1));
    viewer.querySelector('.fly-view-open').addEventListener('click', () => leaveTo(viewer.dataset.item));
    setSound(Engine.isOn());
  }

  function enter() {
    if (on || !map) return;
    buildDom();

    const c = map.getCenter();
    restore = {
      center: [c.lng, c.lat],
      zoom: map.getZoom(),
      bearing: map.getBearing(),
      pitch: map.getPitch(),
      fov: getFov(),
      base: baseIndex,
      dragPan: map.dragPan.isEnabled()
    };

    on = true;
    flying = false;
    paused = false;
    pos = { lat: c.lat, lng: c.lng };
    bearing = map.getBearing();
    alt = ALT_START;
    speed = 0;
    throttle = 0;
    hold = false;
    yaw = 0;
    bank = 0;
    bankRate = 0;
    burn = 0;
    pitchTrim = 0;
    keys.clear();
    mouse.in = false;
    hover.card = null;
    hover.hud = false;
    touchHold = false;
    armed = false;

    document.body.classList.add('flying');

    // Full screen is asked for, never depended on: iOS refuses it outright and
    // the mode is perfectly good without it.
    if (document.documentElement.requestFullscreen && !document.fullscreenElement) {
      document.documentElement.requestFullscreen({ navigationUI: 'hide' }).catch(() => {});
    }

    map.dragPan.disable();
    map.scrollZoom.disable();
    map.doubleClickZoom.disable();
    map.keyboard.disable();
    map.touchZoomRotate.disable();
    map.dragRotate.disable();

    // The engine starts here, inside the click that opened the mode, because
    // that is the gesture the browser wants before it will let a page make a
    // sound. Started later, from a timer, it would stay silent.
    Engine.start();

    const start = () => {
      fitFrame();
      disarm();
      restore.hidden = hideOverlays();

      // Terrain off, and this was measured rather than assumed. At a pitch in
      // the seventies the DEM mesh is seen almost edge-on, and since the tiles
      // stop at zoom 14 while the flight sits around 16 to 18, every triangle
      // is stretched over five zoom levels: the near half of the picture
      // becomes vertical smears of colour. Turning it off gave a clean
      // photograph of the moshava all the way to the horizon.
      //
      // Nothing is lost. Real relief here is 46 m across the whole moshava,
      // which at flying height is invisible; the sense of three dimensions
      // comes from the perspective, the motion, and the photos standing up out
      // of the ground - none of which need a height field.
      map.setTerrain(null);

      addLayers();
      buildWorld();

      const t = clamp((alt - ALT_MIN) / (ALT_MAX - ALT_MIN), 0, 1);
      const pitch = lerp(PITCH_LOW, PITCH_HIGH, Math.sqrt(t));
      const ahead = destination(pos.lat, pos.lng, bearing, alt * Math.tan(pitch * RAD));

      // A take-off rather than a cut. The mode is a change of place as much as
      // a change of controls, and arriving at altitude in one frame reads as a
      // glitch where a rise reads as leaving the ground.
      map.easeTo({
        center: [ahead.lng, ahead.lat],
        zoom: zoomFor(alt, pitch, ahead.lat),
        bearing,
        pitch,
        duration: 1700,
        essential: true
      });
      setTimeout(() => {
        if (!on) return;
        flying = true;
        last = performance.now();
        tick = 0;
      }, 1750);

      intro.hidden = false;
      introTimer = setTimeout(dismissIntro, 9000);
      last = performance.now();
      raf = requestAnimationFrame(step);
    };

    // Satellite is not a preference here, it is the material: the mode is a
    // flight over a photograph of the place.
    if (baseIndex !== 1) {
      map.once('style.load', () => setTimeout(start, 60));
      setBasemap(1);
    } else {
      setTimeout(start, 30);
    }

    addEventListener('keydown', onKeyDown);
    addEventListener('keyup', onKeyUp);
    addEventListener('blur', onBlur);
    // Bound to the window and not to our own overlay: the map canvas sits on
    // top of everything we draw except the cards and the HUD, so a pointer
    // over open ground never reaches an element of ours. The handlers filter
    // by target instead, which is also what keeps a tap on a photo from
    // steering.
    addEventListener('pointermove', onPointerMove);
    addEventListener('pointerdown', onPointerDown);
    addEventListener('pointerup', onPointerUp);
    addEventListener('pointercancel', onPointerUp);
    addEventListener('click', onClick);
    addEventListener('contextmenu', onContextMenu);
    document.documentElement.addEventListener('mouseleave', onPointerLeave);
    addEventListener('touchstart', onTouchStart, { passive: true });
    addEventListener('touchmove', onTouchMove, { passive: true });
    addEventListener('touchend', onTouchEnd);
    addEventListener('resize', onResize);
    document.addEventListener('fullscreenchange', onFullscreen);
    document.addEventListener('visibilitychange', onVisibility);
  }

  function onResize() {
    if (!on) return;
    fitFrame();
  }

  function onFullscreen() {
    // Leaving full screen with F11 or the browser's own gesture means leaving
    // the mode; staying in a hidden-chrome flight the user just dismissed
    // would be the app arguing with them.
    if (on && !document.fullscreenElement) exit();
  }

  function exit(opts = {}) {
    if (!on) return;
    on = false;
    flying = false;
    cancelAnimationFrame(raf);
    raf = null;
    clearTimeout(introTimer);
    Engine.stop();

    removeEventListener('keydown', onKeyDown);
    removeEventListener('keyup', onKeyUp);
    removeEventListener('blur', onBlur);
    removeEventListener('pointermove', onPointerMove);
    removeEventListener('pointerdown', onPointerDown);
    removeEventListener('pointerup', onPointerUp);
    removeEventListener('pointercancel', onPointerUp);
    removeEventListener('click', onClick);
    removeEventListener('contextmenu', onContextMenu);
    document.documentElement.removeEventListener('mouseleave', onPointerLeave);
    removeEventListener('touchstart', onTouchStart);
    removeEventListener('touchmove', onTouchMove);
    removeEventListener('touchend', onTouchEnd);
    removeEventListener('resize', onResize);
    document.removeEventListener('fullscreenchange', onFullscreen);
    document.removeEventListener('visibilitychange', onVisibility);

    for (const [, node] of cards) node.remove();
    cards.clear();
    for (const [, node] of chips) node.remove();
    chips.clear();
    if (viewer) closeShot();

    map.getContainer().style.transform = '';
    worldEl.style.transform = '';
    sky.style.transform = '';
    document.body.classList.remove('flying', 'fly-paused', 'fly-burn', 'fly-unarmed');
    document.documentElement.style.removeProperty('--fly-ox');
    document.documentElement.style.removeProperty('--fly-oy');

    removeLayers();
    if (restore && restore.hidden) showOverlays(restore.hidden);

    map.dragPan.enable();
    map.scrollZoom.enable();
    map.doubleClickZoom.enable();
    map.keyboard.enable();
    map.touchZoomRotate.enable();
    map.dragRotate.enable();

    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});

    const back = restore;
    restore = null;
    setFov(back.fov);
    if (map.setPixelRatio) map.setPixelRatio(undefined);   // back to the device's own
    map.resize();

    // What you flew to is the find, so the position stays; everything else -
    // the base map, the tilt, the zoom - goes back to how it was, because
    // those were choices made before the flight and not by it.
    const land = () => map.easeTo({
      center: [pos.lng, pos.lat],
      zoom: Math.max(back.zoom, 15.5),
      bearing: back.bearing,
      pitch: back.pitch,
      duration: 900
    });

    if (back.base !== 1) {
      // The style swap rebuilds everything through `applyOverlays`, terrain
      // included. Restoring it here as well would start a round of DEM tile
      // requests that setStyle then aborts a moment later, which MapLibre
      // reports to the console as a bare AbortError with no explanation.
      map.once('style.load', () => setTimeout(land, 60));
      setBasemap(back.base);
    } else {
      if (map.getSource('dem')) {
        map.setTerrain({ source: 'dem', exaggeration: TERRAIN_X });
      }
      land();
    }

    if (opts.keepPlace) return;
  }

  const isOn = () => on;
  const toggle = () => (on ? exit() : enter());

  return { enter, exit, toggle, isOn };
})();
