// Extension: the dragon breathes fire. A second button makes the dragon stop and
// pour a wide cone of flame over a large part of the arena for about a second.
// Whatever stands in it keeps losing life. The original game code is not changed:
// three traps in the combat loop read and write the game's own variables, the
// flames are painted over the finished video frame, and the roar is mixed in
// after the SID.

// Addresses in the game (TDA release). X is the fighter: 0 light, 1 dark.
const A = {
  frame: 0x9485,      // once per combat frame, before either fighter is handled
  fighter: 0x94b3,    // per fighter, before its controls (or the computer) are read
  idle: 0x9528,       // the same loop, after the controls: "did not move this frame"
  type: 0xbf29,       // piece type; 0x0d is the dragon, 6 the phoenix
  shot: 0xbd03,       // nonzero while the fighter's shot (or fire ball, scream) is out
  life: 0xbd05,
  reload: 0xbd21,     // frames until the fighter may shoot again
  x: 0xbd3e, y: 0xbd46,   // sprite x = 2 * value + 24, sprite y = value + 50
  pose: 0xbce3,
  over: 0xbcf2,       // set to 0x40 when a fighter dies
  computer: 0xbcc0, computerSide: 0xbd26,
  sound: 0xbf08, soundPos: 0xbf0b, hitTune: 0x95f4,
};
// The bytes the traps sit on; if they differ this is another build of the game.
const EXPECT = [[0x9485, [0x20, 0xb1, 0xa0, 0xa2, 0x01]], [0x94b3, [0xbd, 0xf8, 0xbc]], [0x9528, [0xad, 0x0d, 0xbd]], [0x9a4c, [0xb9, 0x05, 0xbd, 0x38, 0xfd, 0x07, 0xbd]]];
const DRAGON = 0x0d, PHOENIX = 6;

export const BREATH = {
  frames: 72,         // how long one breath lasts
  cooldown: 200,      // frames from the end of a breath until the next one
  reach: 200,         // length of the cone, pixels
  grow: 14,           // frames for the flame to reach full length
  spread: 0.40,       // half width gained per pixel of length
  mouth: 5,           // half width at the mouth
  every: 5,           // a point of life is burnt off every this many frames in the fire
  after: 45,          // the ordinary fireball's reload once the breath ends
};

const DIRS = { 1: [0, -1], 2: [0, 1], 4: [-1, 0], 8: [1, 0], 5: [-1, -1], 9: [1, -1], 6: [-1, 1], 10: [1, 1] };
const unit = ([x, y]) => { const l = Math.hypot(x, y) || 1; return [x / l, y / l]; };
// Poses the game uses for shooting and walking, by direction.
const firePose = ([x, y]) => (y === 0 ? (x < 0 ? 0x16 : 0x0e) : x === 0 ? (y < 0 ? 0x0c : 0x10) : (y < 0 ? 0x0d : 0x0f) + (x < 0 ? 8 : 0));
const walkPose = ([x, y]) => (x < 0 ? 0x11 : x > 0 ? 0x00 : y < 0 ? 0x08 : 0x04);

// Smooth noise for the flame's shape.
const hash = (x, y) => { let n = (x * 374761393 + y * 668265263) | 0; n = (n ^ (n >>> 13)) * 1274126177 | 0; return ((n ^ (n >>> 16)) >>> 0) / 4294967296; };
function noise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y), fx = x - xi, fy = y - yi;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const a = hash(xi, yi), b = hash(xi + 1, yi), c = hash(xi, yi + 1), d = hash(xi + 1, yi + 1);
  return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
}

export function installBreath(machine) {
  const ram = machine.ram;
  for (const [addr, bytes] of EXPECT) for (let i = 0; i < bytes.length; i++) if (ram[addr + i] !== bytes[i]) return null;

  const st = {
    active: 0,          // frames of breath left
    wait: 0,            // frames until the next breath is allowed
    dir: [-1, 0],       // direction of the current breath (grid direction)
    facing: [-1, 0],
    dragon: -1,         // which fighter is the dragon, -1 outside combat
    lastFrame: -10, lastX: 0, lastY: 0,
    asked: [0, 0],      // per joystick port: frames a button press stays wanted
    burns: 0,           // points of life taken, for tests
    roar: 0,
  };
  const centre = (f) => [2 * ram[A.x + f] + 44, ram[A.y + f] + 60 - machine.model.firstLine];   // in frame buffer pixels
  const age = () => BREATH.frames - st.active;
  const reachNow = () => BREATH.reach * Math.min(1, (age() + 1) / BREATH.grow);

  // Where a point lies in the cone: distance along it and sideways share (0 centre, 1 edge).
  function inCone(px, py, margin = 0) {
    const [ox, oy] = centre(st.dragon), [ux, uy] = unit(st.dir);
    const dx = px - ox, dy = py - oy;
    const s = dx * ux + dy * uy, p = Math.abs(dx * uy - dy * ux);
    return s > 4 - margin && s < reachNow() + margin && p < BREATH.mouth + s * BREATH.spread + margin;
  }

  function start(dir) {
    st.dir = dir; st.active = BREATH.frames; st.asked[0] = st.asked[1] = 0;
    ram[A.pose + st.dragon] = firePose(dir);
  }

  function burn(victim) {
    if (ram[A.type + victim] === PHOENIX && ram[A.shot + victim]) return;   // a phoenix in flames does not burn
    const d = st.dragon, life = ram[A.life + victim] - 1;
    ram[A.life + victim] = Math.max(0, life);
    st.burns++;
    if (life <= 0) {
      // What the game does when a shot kills ($9A5C).
      if (ram[A.type + victim] === 0x0e) { ram[0x47fa + victim] = 0x0f; ram[A.shot + victim] = 0; }
      ram[A.y + victim] = 0xce; ram[A.over] = 0x40;
    }
    // The game's own "hit" sound on the dragon's voice.
    ram[A.sound + d] = 0x83; ram[A.soundPos + d] = 0;
    ram[0x3d + d * 2] = ram[A.hitTune + d * 2]; ram[0x3e + d * 2] = ram[A.hitTune + d * 2 + 1];
  }

  machine.trap(A.frame, () => {
    const frame = machine.vic.frames;
    const fresh = frame - st.lastFrame > 30;
    st.lastFrame = frame;
    const d = ram[A.type] === DRAGON ? 0 : ram[A.type + 1] === DRAGON ? 1 : -1;
    st.dragon = d;
    if (d < 0) { st.active = 0; return; }
    const e = d ^ 1, x = ram[A.x + d], y = ram[A.y + d];
    if (fresh) { st.active = 0; st.wait = 60; st.facing = [ram[A.x + e] < x ? -1 : 1, 0]; st.lastX = x; st.lastY = y; }
    if ((ram[A.over] & 0x40) || !ram[A.life + d] || !ram[A.life + e]) { st.active = 0; return; }
    if (x !== st.lastX || y !== st.lastY) st.facing = [Math.sign(x - st.lastX), Math.sign(y - st.lastY)];
    st.lastX = x; st.lastY = y;
    for (let p = 0; p < 2; p++) if (st.asked[p]) st.asked[p]--;

    if (st.active) {
      st.active--;
      if (ram[A.reload + d] < 3) ram[A.reload + d] = 3;       // no fireballs meanwhile
      const [ex, ey] = centre(e);
      if (age() % BREATH.every === 0 && inCone(ex, ey, 7)) burn(e);
      if (!st.active) { st.wait = BREATH.cooldown; ram[A.reload + d] = BREATH.after; ram[A.pose + d] = walkPose(st.dir); }
      return;
    }
    if (st.wait) { st.wait--; return; }
    if (ram[A.computer] && ram[A.computerSide] === d) {
      // The computer breathes when the enemy is well inside the reach, towards it.
      const [ox, oy] = centre(d), [ex, ey] = centre(e), dx = ex - ox, dy = ey - oy;
      if (Math.hypot(dx, dy) > BREATH.reach * 0.8 || hash(frame, 7) > 0.06) return;
      const a = Math.round(Math.atan2(dy, dx) / (Math.PI / 4));
      start([Math.round(Math.cos(a * Math.PI / 4)), Math.round(Math.sin(a * Math.PI / 4))]);
    } else if (st.asked[d]) {
      start(DIRS[machine.joy[d] & 15] || st.facing);
    }
  });

  // While breathing the dragon stands still: skip its controls for this frame.
  machine.trap(A.fighter, () => { if (st.active && machine.cpu.x === st.dragon) machine.cpu.pc = A.idle; });

  // The flames, painted over the finished frame in the C64's own colours.
  const WIDTH = 384, LEFT = 32;
  machine.vic.onFrame = () => {
    st.roar = st.active && machine.vic.frames - st.lastFrame < 3 ? 1 : 0;
    if (!st.roar) return;
    const fb = machine.vic.fb, m = machine.model, top = 51 - m.firstLine, bottom = top + 200;
    const [ox, oy] = centre(st.dragon), [ux, uy] = unit(st.dir);
    const reach = reachNow(), t = machine.vic.frames;
    const fade = Math.min(1, st.active / 10);
    const endW = BREATH.mouth + reach * BREATH.spread;
    // Bounding box of the cone.
    const xs = [ox, ox + ux * reach - uy * endW, ox + ux * reach + uy * endW], ys = [oy, oy + uy * reach + ux * endW, oy + uy * reach - ux * endW];
    const x0 = Math.max(LEFT, Math.floor(Math.min(...xs) - 8) & ~1), x1 = Math.min(LEFT + 320, Math.ceil(Math.max(...xs) + 8));
    const y0 = Math.max(top, Math.floor(Math.min(...ys) - 8)), y1 = Math.min(bottom, Math.ceil(Math.max(...ys) + 8));
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x += 2) {       // double-wide pixels, like the game's sprites
        const dx = x + 1 - ox, dy = y - oy;
        const s = dx * ux + dy * uy;
        if (s < 6 || s > reach + 6) continue;
        const w = BREATH.mouth + s * BREATH.spread, p = (dx * uy - dy * ux) / w;
        if (p < -1.25 || p > 1.25) continue;
        const along = s / reach;
        const n = noise(s * 0.055 - t * 0.42, p * 2.2 + t * 0.05) * 0.65 + noise(s * 0.16 - t * 0.9, p * 5 + 9) * 0.35;
        let heat = (1 - along * along * 0.75) * (1 - p * p) * 1.15 + (n - 0.5) * 0.95;
        heat *= fade;
        if (along > 0.85) heat -= (along - 0.85) * 3;
        if (heat < 0.2) continue;
        const c = heat > 0.92 ? 1 : heat > 0.66 ? 7 : heat > 0.42 ? 8 : 2;
        if (c === 2 && (((x >> 1) + y) & 1)) continue;       // the cool fringe is see-through
        const i = y * WIDTH + x;
        fb[i] = c; fb[i + 1] = c;
      }
    }
  };

  // The roar: filtered noise, mixed in after the SID.
  let level = 0, lo = 0, rumble = 0, seed = 0x1234567, phase = 0;
  const rate = machine.sid.sampleRate;
  const kLo = 1 - Math.exp(-2 * Math.PI * 1100 / rate), kRumble = 1 - Math.exp(-2 * Math.PI * 140 / rate);
  const up = 1 / (0.06 * rate), down = 1 / (0.3 * rate);
  machine.sid.extra = () => {
    if (st.roar) { if (level < 1) level = Math.min(1, level + up); } else if (level > 0) level = Math.max(0, level - down);
    else return 0;
    seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
    const white = (seed >>> 0) / 2147483648 - 1;
    lo += (white - lo) * kLo; rumble += (white - rumble) * kRumble;
    phase += 23 / rate;
    const flutter = 0.8 + 0.2 * Math.sin(phase * 2 * Math.PI);
    return (lo * 0.55 + rumble * 1.6) * level * level * flutter * 0.9;
  };

  return {
    state: st,
    // A second-button press on a joystick port (0 = port 1 / light, 1 = port 2 / dark).
    press(port) { st.asked[port] = 12; },
    get ready() { return st.dragon >= 0 && !st.active && !st.wait; },
  };
}

// The game unpacks itself after RUN, so wait until its combat code is in memory.
export function armBreath(machine, onReady) {
  machine.vic.onFrame = () => {
    const ext = installBreath(machine);
    if (ext && onReady) onReady(ext);
  };
}
