/* מצב תעופה - flying over the moshava to find its shortcuts.
 *
 * The rest of the app answers "where is the shortcut I already know about".
 * This answers the other question: what is out there. You take off over the
 * satellite image, steer with the keyboard, and trails light up as you come
 * near them - nothing is drawn until you are close enough to have found it.
 * The photos that are attached to a trail hang in the air above it and grow
 * as you close in, so a picture is a thing you fly towards.
 *
 * Built on the map that is already here rather than on a 3D engine. MapLibre
 * already holds the satellite raster, the terrain, and a camera that tilts to
 * 80 degrees; what it does not have is a flight model. So the whole of this
 * file is really two things: a camera that thinks in metres above the ground,
 * and a reveal rule that decides what you have discovered.
 *
 * The camera is honest. `alt` is a real height in metres, and the zoom handed
 * to the map is derived from it through MapLibre's own projection geometry
 * (see `zoomFor`). That is what makes the photos grow correctly on approach -
 * they are sized in world metres and projected, not scaled by a fudge factor.
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
  const PITCH_MIN = 58;      // limits for the mouse-drag trim
  const PITCH_MAX = 80;

  const SPEED_K = 0.45;      // cruise m/s per metre of altitude
  const SPEED_MIN = 14;
  const SPEED_MAX = 110;
  const BOOST = 2.1;
  const ACCEL = 1.9;         // how fast speed approaches its target, per second
  const YAW_RATE = 62;       // degrees per second at full deflection
  const YAW_ACCEL = 5.0;
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

  const NEAR_ROLL = 6;       // biggest bank angle we will ever ask for
  const OVER = 8;            // per-cent the map is grown by, to cover the bank

  const RAD = Math.PI / 180;
  const M_PER_DEG_LAT = 111320;

  /* MapLibre's default field of view is 36.87 degrees, and it places the
   * camera `0.5 / tan(fov/2) * height` pixels from the centre point - which
   * for that fov is exactly 1.5 heights. Both the altitude-to-zoom conversion
   * and the horizon line fall out of this one number. */
  const FOCAL = 1.5;

  /* ---------- state ---------- */

  let on = false;
  let flying = false;        // false during the take-off ease, true after
  let raf = null;
  let last = 0;
  let tick = 0;              // seconds until the next reveal recompute

  const keys = new Set();
  let pos = { lat: 0, lng: 0 };
  let alt = ALT_START;
  let bearing = 0;
  let speed = 0;
  let yaw = 0;               // current turn rate, degrees/second
  let roll = 0;              // visual bank only, MapLibre 4 has no roll axis
  let maxRoll = 0;
  let pitchTrim = 0;         // mouse-drag adjustment on top of the auto pitch

  let world = [];            // every item near enough to ever matter
  let shots = [];            // every photo, placed on the ground
  let seen = new Map();      // item id -> when it came into range, for the pulse
  let near = [];             // last reveal result, reused between recomputes
  let nearest = null;        // the closest revealed item, for Enter
  let paused = false;        // a photo is open full-size

  let restore = null;        // what to put back on the way out

  const cards = new Map();   // photo key -> element, pooled across frames
  const chips = new Map();   // item id -> element

  let root = null, sky = null, band = null, hazeEl = null, worldEl = null;
  let elAlt = null, elSpeed = null, elName = null, elHint = null, rushEl = null;
  let intro = null, viewer = null;

  const el = (id) => document.getElementById(id);
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const lerp = (a, b, t) => a + (b - a) * t;

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
   *  MapLibre holds the camera `FOCAL * height` pixels from the centre point,
   *  along the view axis; the vertical leg of that is `cos(pitch)` of it. So
   *  the metres-per-pixel we need is alt / (that leg), and the zoom is what
   *  gives that scale at this latitude. Inverting the projection like this,
   *  rather than picking zooms by eye, is what lets the altitude readout mean
   *  something and the photos size themselves in real metres. */
  function zoomFor(a, pitch, lat) {
    const h = map.getContainer().clientHeight || 800;
    const mpp = a / (FOCAL * h * Math.cos(pitch * RAD));
    const worldPx = 40075016.686 * Math.cos(lat * RAD) / mpp;
    return clamp(Math.log2(worldPx / 512), 1, 22);
  }

  /** Screen y of the horizon, in map-container pixels. Points at infinity sit
   *  `FOCAL / tan(pitch)` heights above the centre of the view. */
  function horizonY(pitch) {
    const h = map.getContainer().clientHeight || 800;
    return h * (0.5 - FOCAL / Math.tan(pitch * RAD));
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

      const photos = (it.photos || []).filter((p) => p && (p.thumb || p.full));
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
    node.className = 'fly-card';
    node.innerHTML =
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
    // is wider than the screen by the bank allowance, so its centre and the
    // screen's centre coincide but its edges are outside.
    const cw = map.getContainer().clientWidth;
    const padX = (cw - innerWidth) / 2;
    const padY = ((map.getContainer().clientHeight || vh) - innerHeight) / 2;

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

      const w = clamp(wMetres * ppm, CARD_MIN_PX, vh * CARD_MAX_VH);
      const tether = clamp(hMetres * ppm * sinP, 6, vh * 0.22);
      const cardH = w * 0.75 + 26;

      // Off the screen is off the screen. The heading test above only rejects
      // what is behind the camera; a photo a little to the side and close by
      // passes it and still lands two thousand pixels below the bottom edge,
      // because at this tilt the ground under you is not in the picture. Those
      // were counting against the ten slots and showing nothing.
      if (p.x + w / 2 < padX - 40 || p.x - w / 2 > padX + innerWidth + 40) continue;
      if (p.y - tether - cardH > padY + innerHeight + 40) continue;

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
      node.style.left = `${p.x}px`;
      node.style.top = `${p.y - tether}px`;
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
      node.style.left = `${p.x}px`;
      node.style.top = `${p.y}px`;
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

    if (!paused) {
      const fwd = (keys.has('KeyW') ? 1 : 0) - (keys.has('KeyS') ? 1 : 0);
      const turn = (keys.has('KeyD') ? 1 : 0) - (keys.has('KeyA') ? 1 : 0)
        + (keys.has('ArrowRight') ? 1 : 0) - (keys.has('ArrowLeft') ? 1 : 0);
      const climb = (keys.has('ArrowUp') ? 1 : 0) - (keys.has('ArrowDown') ? 1 : 0);
      const boost = keys.has('ShiftLeft') || keys.has('ShiftRight') ? BOOST : 1;

      const cruise = clamp(alt * SPEED_K, SPEED_MIN, SPEED_MAX) * boost;
      const want = fwd > 0 ? cruise : fwd < 0 ? -cruise * 0.4 : 0;
      speed += (want - speed) * clamp(ACCEL * dt, 0, 1);

      const wantYaw = clamp(turn, -1, 1) * YAW_RATE;
      yaw += (wantYaw - yaw) * clamp(YAW_ACCEL * dt, 0, 1);
      bearing = (bearing + yaw * dt + 360) % 360;

      // Climb rate is a fraction of the height you are at, so the whole range
      // from rooftop to survey takes a handful of seconds either way instead
      // of being brisk at the bottom and interminable at the top.
      if (climb) alt = clamp(alt * (1 + climb * CLIMB_K * dt) + climb * 6 * dt,
                             ALT_MIN, ALT_MAX);

      if (speed) {
        const to = destination(pos.lat, pos.lng, bearing, speed * dt);
        pos = to;
      }
    } else {
      speed += (0 - speed) * clamp(3 * dt, 0, 1);
    }

    // Bank into the turn. MapLibre 4 has no roll axis, so the map, the sky and
    // the floating cards are rotated together in CSS - they have to move as
    // one or the photos slide off their trails.
    const wantRoll = maxRoll ? clamp(-yaw / YAW_RATE, -1, 1) * maxRoll : 0;
    roll += (wantRoll - roll) * clamp(3.2 * dt, 0, 1);
    if (maxRoll) {
      const tf = `rotate(${roll.toFixed(2)}deg)`;
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
  }

  function paintHud(pitch) {
    // The sky is a band that ends exactly at the horizon, so the pale end of
    // its gradient always meets the ground rather than landing wherever a
    // fixed gradient happened to put it. It sits behind the map, so the seam
    // itself is never seen; the haze in front of the map covers where the far
    // tiles stop.
    const y = horizonY(pitch);
    hazeEl.style.top = `${y}px`;
    band.style.height = `${Math.max(y + 6, 0)}px`;

    elAlt.textContent = `${Math.round(alt)} מ׳`;
    elSpeed.textContent = `${Math.round(Math.abs(speed) * 3.6)} קמ״ש`;

    const cruise = clamp(alt * SPEED_K, SPEED_MIN, SPEED_MAX) * BOOST;
    rushEl.style.opacity = clamp((Math.abs(speed) / cruise - 0.34) * 0.62, 0, 0.34);

    if (nearest && nearest.name) {
      elName.textContent = nearest.name;
      elName.hidden = false;
      elHint.hidden = false;
    } else {
      elName.hidden = true;
      elHint.hidden = true;
    }
  }

  /* ---------- looking at one photo ---------- */

  function openShot(shot) {
    paused = true;
    // The instruments go while a picture is being looked at. They report a
    // flight that is standing still, and their close button sits in the same
    // corner as the viewer's own - two crosses on top of each other, neither
    // of them obviously the one that closes what is in front of you.
    document.body.classList.add('fly-paused');
    viewer.querySelector('.fly-view-img').src = shot.full;
    viewer.querySelector('.fly-view-name').textContent = shot.name;
    const cap = viewer.querySelector('.fly-view-cap');
    cap.textContent = shot.cap || '';
    cap.hidden = !shot.cap;
    viewer.dataset.item = shot.owner.id;
    viewer.hidden = false;
  }

  function closeShot() {
    document.body.classList.remove('fly-paused');
    viewer.hidden = true;
    viewer.querySelector('.fly-view-img').src = '';
    paused = false;
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
    if (e.code === 'Enter') {
      e.preventDefault();
      openNearest();
      return;
    }
    if (e.code === 'KeyH' || e.code === 'Slash') { e.preventDefault(); toggleIntro(); return; }
    if (e.code === 'KeyR') { e.preventDefault(); pitchTrim = 0; return; }
    if (/^(Key[WASD]|Arrow(Up|Down|Left|Right)|Shift(Left|Right)|Space)$/.test(e.code)) {
      e.preventDefault();
      keys.add(e.code);
      dismissIntro();
    }
  }

  function onKeyUp(e) {
    keys.delete(e.code);
  }

  let drag = null;

  function onDown(e) {
    if (e.target.closest('.fly-card, .fly-hud, .fly-view')) return;
    drag = { x: e.clientX, y: e.clientY, moved: 0 };
  }

  function onMove(e) {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    drag.moved += Math.abs(dx) + Math.abs(dy);
    // Dragging looks around: sideways swings the heading, up and down trims
    // the tilt away from what altitude picked for you.
    bearing = (bearing - dx * 0.22 + 360) % 360;
    pitchTrim = clamp(pitchTrim + dy * 0.12, -18, 8);
    dismissIntro();
  }

  function onUp() { drag = null; }

  /* Touch: no keyboard to fly with, so a drag steers and holding the screen
   * is the throttle. Deliberately small - this mode is a desktop pleasure and
   * a phone should get something that works rather than a second control
   * scheme to learn. */
  let touch = null;

  function onTouchStart(e) {
    if (e.target.closest('.fly-card, .fly-hud, .fly-view')) return;
    const t = e.touches[0];
    touch = { x: t.clientX, y: t.clientY };
    keys.add('KeyW');
    dismissIntro();
  }

  function onTouchMove(e) {
    if (!touch) return;
    const t = e.touches[0];
    const dx = t.clientX - touch.x, dy = t.clientY - touch.y;
    touch.x = t.clientX; touch.y = t.clientY;
    bearing = (bearing - dx * 0.28 + 360) % 360;
    if (Math.abs(dy) > 1) alt = clamp(alt * (1 + dy * 0.004), ALT_MIN, ALT_MAX);
  }

  function onTouchEnd() { touch = null; keys.delete('KeyW'); }

  function openNearest() {
    if (!nearest) return;
    const id = nearest.id;
    exit({ keepPlace: true });
    setTimeout(() => select(id), 320);
  }

  /* ---------- intro ---------- */

  let introTimer = null;

  function dismissIntro() {
    if (!intro || intro.hidden) return;
    intro.hidden = true;
    clearTimeout(introTimer);
  }

  function toggleIntro() {
    intro.hidden = !intro.hidden;
    clearTimeout(introTimer);
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
      <div class="fly-vig"></div>
      <div class="fly-hud">
        <div class="fly-readout">
          <span class="fly-gauge"><b id="fly-alt">—</b><i>גובה</i></span>
          <span class="fly-gauge"><b id="fly-speed">—</b><i>מהירות</i></span>
        </div>
        <button class="fly-x" id="fly-x" aria-label="יציאה ממצב תעופה">&times;</button>
        <button class="fly-help" id="fly-help" aria-label="מקשים">?</button>
        <div class="fly-found">
          <strong id="fly-name" hidden></strong>
          <span id="fly-hint" class="fly-hint" hidden><kbd>Enter</kbd> <span>לפתיחה</span></span>
        </div>
        <p class="fly-credit">Esri · Maxar · Earthstar Geographics · גובה: Mapzen / AWS</p>
      </div>
      <div class="fly-intro" id="fly-intro">
        <div class="fly-intro-card">
          <h2>מצב תעופה</h2>
          <p>אתה מרחף מעל פרדס חנה־כרכור. דרכי הקיצור נדלקות כשמתקרבים אליהן,
             והתמונות שלהן תלויות באוויר מעל השביל.</p>
          <ul class="fly-keys">
            <li><kbd>W</kbd><kbd>S</kbd><span>קדימה ואחורה</span></li>
            <li><kbd>A</kbd><kbd>D</kbd><span>פנייה</span></li>
            <li><kbd>↑</kbd><kbd>↓</kbd><span>גובה</span></li>
            <li><kbd>Shift</kbd><span>האצה</span></li>
            <li><kbd>Enter</kbd><span>פתיחת השביל הקרוב</span></li>
            <li><kbd>Esc</kbd><span>יציאה</span></li>
          </ul>
          <p class="fly-intro-foot">גרירה עם העכבר מסתכלת מסביב. לחיצה על תמונה פותחת אותה.</p>
        </div>
      </div>
      <div class="fly-view" id="fly-view" hidden>
        <button class="fly-view-x" aria-label="סגירה">&times;</button>
        <img class="fly-view-img" alt="" referrerpolicy="no-referrer">
        <div class="fly-view-foot">
          <strong class="fly-view-name"></strong>
          <span class="fly-view-cap"></span>
          <button class="fly-view-open">פתח את השביל</button>
        </div>
      </div>`;
    document.body.appendChild(root);

    sky = el('fly-sky');
    band = el('fly-band');
    worldEl = el('fly-world');
    hazeEl = el('fly-haze');
    rushEl = el('fly-rush');
    elAlt = el('fly-alt');
    elSpeed = el('fly-speed');
    elName = el('fly-name');
    elHint = el('fly-hint');
    intro = el('fly-intro');
    viewer = el('fly-view');

    el('fly-x').addEventListener('click', () => exit());
    el('fly-help').addEventListener('click', toggleIntro);
    intro.addEventListener('click', dismissIntro);
    viewer.querySelector('.fly-view-x').addEventListener('click', closeShot);
    viewer.querySelector('.fly-view-open').addEventListener('click', () => {
      const id = viewer.dataset.item;
      closeShot();
      exit({ keepPlace: true });
      setTimeout(() => select(id), 320);
    });
  }

  /** How far we may bank before the rotated map stops covering the screen.
   *
   *  The map is grown by OVER per cent on every side; a rectangle rotated by
   *  θ still covers the viewport while `cos θ + r sin θ <= 1 + 2*OVER/100`,
   *  where r is the long-to-short side ratio. Solved rather than guessed,
   *  because a guess shows a black wedge in the corner on some screen. */
  function rollLimit() {
    const w = innerWidth, h = innerHeight;
    if (!matchMedia('(pointer: fine)').matches) return 0;
    const r = Math.max(w / h, h / w);
    const k = 1 + (2 * OVER) / 100;
    const hyp = Math.hypot(1, r);
    if (k / hyp > 1) return NEAR_ROLL;
    const theta = (Math.asin(k / hyp) - Math.atan2(1, r)) / RAD;
    return theta < 0.8 ? 0 : Math.min(theta, NEAR_ROLL);
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
    yaw = 0;
    roll = 0;
    pitchTrim = 0;
    keys.clear();

    maxRoll = rollLimit();
    document.documentElement.style.setProperty('--fly-over', maxRoll ? `${OVER}%` : '0%');
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

    const start = () => {
      map.resize();
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
    addEventListener('blur', () => keys.clear());
    // Bound to the window and not to our own overlay: the map canvas sits on
    // top of everything we draw except the cards and the HUD, so a drag over
    // open ground never reaches an element of ours. `onDown` filters by target
    // instead, which is also what keeps a tap on a photo from steering.
    addEventListener('pointerdown', onDown);
    addEventListener('pointermove', onMove);
    addEventListener('pointerup', onUp);
    addEventListener('touchstart', onTouchStart, { passive: true });
    addEventListener('touchmove', onTouchMove, { passive: true });
    addEventListener('touchend', onTouchEnd);
    addEventListener('resize', onResize);
    document.addEventListener('fullscreenchange', onFullscreen);
  }

  function onResize() {
    if (!on) return;
    maxRoll = rollLimit();
    document.documentElement.style.setProperty('--fly-over', maxRoll ? `${OVER}%` : '0%');
    map.resize();
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

    removeEventListener('keydown', onKeyDown);
    removeEventListener('keyup', onKeyUp);
    removeEventListener('pointermove', onMove);
    removeEventListener('pointerup', onUp);
    removeEventListener('resize', onResize);
    document.removeEventListener('fullscreenchange', onFullscreen);
    removeEventListener('pointerdown', onDown);
    removeEventListener('touchstart', onTouchStart);
    removeEventListener('touchmove', onTouchMove);
    removeEventListener('touchend', onTouchEnd);

    for (const [, node] of cards) node.remove();
    cards.clear();
    for (const [, node] of chips) node.remove();
    chips.clear();
    if (viewer) closeShot();

    map.getContainer().style.transform = '';
    worldEl.style.transform = '';
    sky.style.transform = '';
    document.body.classList.remove('flying', 'fly-paused');
    document.documentElement.style.removeProperty('--fly-over');

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
