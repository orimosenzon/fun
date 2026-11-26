/* The F-16's glass and dials: everything in the cockpit that is drawn with a
 * pen rather than built (26/11/2026). No three.js in here: f16_pit.js holds
 * the canvases this file draws on and hangs them in the cockpit.
 *
 * Three things are drawn:
 *
 *   drawHud      the head-up display, on a square canvas that f16_pit.js
 *                hangs forty metres ahead of the aircraft and shows only
 *                through the combiner glass, which is how a real HUD works:
 *                the symbols are focused at infinity, they stay on the world
 *                when the head moves, and the glass is a window onto them.
 *   drawPanel    the instrument panel's face, once: bezels, legends, screws.
 *   drawDyn      everything on the panel that moves, onto one atlas canvas:
 *                the two MFDs, the DED, the attitude ball, the HSI, the
 *                airspeed and altitude dials, the engine column, the fuel,
 *                the tapes, the indexers and the warning lights.
 *
 * The layout follows the F-16C block 50's as the public manuals and the
 * simulators show it: HUD over the ICP with the threat scope to its left and
 * the DED to its right; an MFD at each knee; between them the attitude
 * indicator with the angle-of-attack and vertical-velocity tapes at its
 * sides, the airspeed and the altimeter above it and the HSI below; the
 * engine column (oil, nozzle, RPM, FTIT) right of the right MFD; the gear
 * handle and its three greens at the lower left.
 */
import { qrot } from './f16_core.js';

const RAD = Math.PI / 180;
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const pad = (n, w) => String(Math.round(n)).padStart(w, '0');
const MONO = 'ui-monospace, "DejaVu Sans Mono", Menlo, Consolas, monospace';

/* ====================================================================
 * The HUD
 * ==================================================================== */

export const HUD = {
  N: 1024,             // the canvas, pixels
  HALF: 13,            // half the field it covers, degrees
  DEP: 4               // its centre is this far below the waterline
};
const HK = 512 / Math.tan(HUD.HALF * RAD);
const HY0 = Math.tan(-HUD.DEP * RAD);
const GREEN = '#8dff9a';

/** A direction in the aircraft's own axes (forward -z) to canvas pixels, or
 *  null when it is behind the glass. */
function proj(d) {
  if (-d[2] < 0.15) return null;
  return [512 + HK * d[0] / -d[2], 512 - HK * (d[1] / -d[2] - HY0)];
}
/** A fixed place on the glass, in degrees right of the centreline and above
 *  the waterline. */
const at = (az, el) => [512 + HK * Math.tan(az * RAD), 512 - HK * (Math.tan(el * RAD) - HY0)];

const worldDir = (brg, el) => {
  const c = Math.cos(el * RAD);
  return [Math.sin(brg * RAD) * c, Math.sin(el * RAD), -Math.cos(brg * RAD) * c];
};

/** One frame of the HUD. `s`: q (attitude), vel [x,y,z] m/s, cas (kt),
 *  alt (ft), hdg, pitch, roll, track, mach, nz, maxG, aoa (deg), gear (0..1),
 *  ap, limit, over, pull (pull-up cue), stpt { dir: [x,y,z] world unit,
 *  nm, n }, fuelLow, grounded, t (seconds), sb. */
export function drawHud(c, s) {
  const N = HUD.N;
  c.clearRect(0, 0, N, N);
  c.save();
  c.strokeStyle = GREEN; c.fillStyle = GREEN;
  c.lineWidth = 5; c.lineCap = 'round'; c.lineJoin = 'round';
  c.shadowColor = 'rgba(0,45,10,.9)'; c.shadowBlur = 4;
  c.font = `700 36px ${MONO}`; c.textBaseline = 'middle';
  c.direction = 'ltr';
  const qi = [-s.q[0], -s.q[1], -s.q[2], s.q[3]];
  const body = (d) => qrot(qi, d);
  const blink = (s.t * 3 % 1) < 0.55;

  // ---- the pitch ladder, centred on where the aircraft is going ----
  const V = Math.hypot(s.vel[0], s.vel[1], s.vel[2]);
  const psi = V > 25 ? s.track : s.hdg;
  c.save();
  // The ladder keeps clear of the scales at the sides and the heading tape.
  c.beginPath(); c.rect(at(-4.9, 0)[0], at(0, 5.5)[1], at(4.9, 0)[0] - at(-4.9, 0)[0], at(0, -9.6)[1] - at(0, 5.5)[1]); c.clip();
  for (let phi = -90; phi <= 90; phi += 5) {
    if (Math.abs(phi - s.pitch) > 24 && Math.abs(phi) < 85) continue;
    const cs = Math.max(Math.cos(phi * RAD), 0.07);
    const half = (phi === 0 ? 9 : 3.1) / cs, gap = (phi === 0 ? 1.4 : 1.1) / cs;
    const A = proj(body(worldDir(psi - half, phi))), a = proj(body(worldDir(psi - gap, phi)));
    const b = proj(body(worldDir(psi + gap, phi))), B = proj(body(worldDir(psi + half, phi)));
    if (!A || !a || !b || !B) continue;
    if (Math.min(A[1], B[1]) > N + 60 || Math.max(A[1], B[1]) < -60) continue;
    c.setLineDash(phi < 0 ? [20, 13] : []);
    c.beginPath(); c.moveTo(A[0], A[1]); c.lineTo(a[0], a[1]); c.moveTo(b[0], b[1]); c.lineTo(B[0], B[1]); c.stroke();
    c.setLineDash([]);
    if (phi !== 0) {
      // The tips point at the horizon, and the numbers ride the ends.
      const tip = phi > 0 ? -0.75 : 0.75;
      const At = proj(body(worldDir(psi - half, phi + tip))), Bt = proj(body(worldDir(psi + half, phi + tip)));
      if (At && Bt) { c.beginPath(); c.moveTo(A[0], A[1]); c.lineTo(At[0], At[1]); c.moveTo(B[0], B[1]); c.lineTo(Bt[0], Bt[1]); c.stroke(); }
      const ang = Math.atan2(B[1] - A[1], B[0] - A[0]);
      const ux = Math.cos(ang), uy = Math.sin(ang);
      c.font = `700 31px ${MONO}`;
      for (const [P, sg] of [[A, -1], [B, 1]]) {
        c.save(); c.translate(P[0] + sg * 34 * ux, P[1] + sg * 34 * uy); c.rotate(ang);
        c.textAlign = 'center'; c.fillText(String(Math.abs(phi)), 0, 0); c.restore();
      }
    }
  }
  c.restore();
  c.font = `700 36px ${MONO}`;

  // ---- the gun cross: where the nose points ----
  {
    const [x, y] = at(0, 0);
    c.beginPath(); c.moveTo(x - 16, y); c.lineTo(x - 5, y); c.moveTo(x + 5, y); c.lineTo(x + 16, y);
    c.moveTo(x, y - 16); c.lineTo(x, y - 5); c.moveTo(x, y + 5); c.lineTo(x, y + 16); c.stroke();
  }

  // ---- the flight path marker: where it is going ----
  let fpm = null;
  {
    let p = V > 12 ? proj(body([s.vel[0] / V, s.vel[1] / V, s.vel[2] / V])) : at(0, 0);
    const x0 = at(-4.4, 0)[0], x1 = at(4.4, 0)[0], y0 = at(0, 5)[1], y1 = at(0, -9.2)[1];
    let lim = !p;
    if (!p) p = at(0, -9);
    if (p[0] < x0 || p[0] > x1 || p[1] < y0 || p[1] > y1) lim = true;
    p = [clamp(p[0], x0, x1), clamp(p[1], y0, y1)];
    fpm = p;
    const r = 13;
    c.beginPath(); c.arc(p[0], p[1], r, 0, 7);
    c.moveTo(p[0] - r, p[1]); c.lineTo(p[0] - r - 26, p[1]);
    c.moveTo(p[0] + r, p[1]); c.lineTo(p[0] + r + 26, p[1]);
    c.moveTo(p[0], p[1] - r); c.lineTo(p[0], p[1] - r - 14);
    c.stroke();
    if (lim) { c.beginPath(); c.moveTo(p[0] - 9, p[1] - 9); c.lineTo(p[0] + 9, p[1] + 9); c.moveTo(p[0] + 9, p[1] - 9); c.lineTo(p[0] - 9, p[1] + 9); c.stroke(); }
    // Gear down: the angle-of-attack bracket beside the marker. Level with
    // its wing at thirteen degrees, the approach.
    if (s.gear > 0.5) {
      const off = clamp((s.aoa - 13) * 9, -60, 60);
      const bx = p[0] - r - 46, by = p[1] + off;
      c.beginPath(); c.moveTo(bx + 10, by - 26); c.lineTo(bx, by - 26); c.lineTo(bx, by + 26); c.lineTo(bx + 10, by + 26); c.stroke();
    }
  }

  // ---- the steerpoint: a diamond on the ground where the trail is ----
  if (s.stpt && s.stpt.dir) {
    const p = proj(body(s.stpt.dir));
    if (p && p[0] > 90 && p[0] < N - 90 && p[1] > 90 && p[1] < N - 110) {
      c.beginPath(); c.moveTo(p[0], p[1] - 15); c.lineTo(p[0] + 15, p[1]); c.lineTo(p[0], p[1] + 15); c.lineTo(p[0] - 15, p[1]); c.closePath(); c.stroke();
    }
  }

  // ---- airspeed, left ----
  const tape = (az, val, perDeg, minor, major, label, side) => {
    const [x, yc] = at(az, -HUD.DEP);
    const span = 3.6;                                 // degrees above and below the box
    const pxPer = (at(0, 0)[1] - at(0, -1)[1]) * -1;  // pixels per degree, near enough flat here
    const k = pxPer / perDeg;
    c.save();
    c.beginPath(); c.rect(x - 110, yc - span * pxPer, 220, 2 * span * pxPer); c.clip();
    c.font = `700 31px ${MONO}`;
    const lo = Math.floor((val - span * perDeg) / minor) * minor;
    for (let v = lo; v <= val + span * perDeg; v += minor) {
      if (v < 0) continue;
      const y = yc - (v - val) * k;
      if (Math.abs(y - yc) < 27) continue;             // the scale is blanked behind the box
      const big = Math.round(v / minor) % Math.round(major / minor) === 0;
      c.beginPath(); c.moveTo(x, y); c.lineTo(x - side * (big ? 20 : 11), y); c.stroke();
      if (big) { c.textAlign = side > 0 ? 'right' : 'left'; c.fillText(label(v), x - side * 27, y); }
    }
    c.restore();
    return [x, yc];
  };
  {
    const [x, y] = tape(-5.6, s.cas, 22, 10, 50, (v) => pad(v / 10, 2), 1);
    const t = pad(Math.max(0, s.cas), 3);
    c.save(); c.shadowBlur = 0; c.fillStyle = 'rgba(0,0,0,.45)'; c.fillRect(x - 112, y - 23, 94, 46); c.restore();
    c.strokeRect(x - 112, y - 23, 94, 46);
    c.textAlign = 'right'; c.fillText(t, x - 24, y + 1);
    c.beginPath(); c.moveTo(x - 18, y); c.lineTo(x - 4, y); c.stroke();
    c.font = `700 30px ${MONO}`; c.textAlign = 'center'; c.fillText('C', x - 132, y + 1);
    c.font = `700 36px ${MONO}`;
  }
  // ---- altitude, right ----
  {
    const [x, y] = tape(5.6, s.alt, 260, 100, 500, (v) => `${Math.floor(v / 1000)},${Math.round(v % 1000 / 100)}`, -1);
    const a = Math.round(s.alt / 10) * 10;
    const t = a < 1000 ? String(a) : `${Math.floor(a / 1000)},${pad(Math.abs(a % 1000), 3)}`;
    c.save(); c.shadowBlur = 0; c.fillStyle = 'rgba(0,0,0,.45)'; c.fillRect(x + 18, y - 23, 138, 46); c.restore();
    c.strokeRect(x + 18, y - 23, 138, 46);
    c.textAlign = 'right'; c.fillText(t, x + 151, y + 1);
    c.beginPath(); c.moveTo(x + 4, y); c.lineTo(x + 18, y); c.stroke();
    // The radar altimeter, while it can see the ground.
    if (s.ralt < 1500 && Math.abs(s.roll) < 60 && Math.abs(s.pitch) < 60) {
      const [rx, ry] = at(5.6, -8.6);
      c.font = `700 31px ${MONO}`; c.textAlign = 'right';
      c.fillText(`R ${Math.round(s.ralt / 10) * 10}`.replace(/(\d)(\d{3})$/, '$1,$2'), rx + 151, ry);
      c.strokeRect(rx + 18, ry - 20, 138, 40);
      c.font = `700 36px ${MONO}`;
    }
  }

  // ---- heading, along the bottom ----
  {
    const [x, y] = at(0, -11.5);
    const pxPer = 24;                                  // pixels per degree of heading: 2.4 degrees of scale per degree of glass
    c.save();
    c.beginPath(); c.rect(x - 150, y - 44, 300, 70); c.clip();
    c.font = `700 30px ${MONO}`; c.textAlign = 'center';
    const h0 = Math.floor(s.hdg / 5) * 5;
    for (let d = -10; d <= 10; d += 5) {
      const h = h0 + d, xx = x + (h - s.hdg) * pxPer;
      const big = ((h % 10) + 10) % 10 === 0;
      c.beginPath(); c.moveTo(xx, y); c.lineTo(xx, y - (big ? 13 : 7)); c.stroke();
      if (big && Math.abs(xx - x) > 62) c.fillText(pad((((h % 360) + 360) % 360) / 10 || 36, 2), xx, y - 31);
    }
    c.restore();
    c.beginPath(); c.moveTo(x, y + 3); c.lineTo(x - 8, y + 16); c.lineTo(x + 8, y + 16); c.closePath(); c.stroke();
    c.save(); c.shadowBlur = 0; c.fillStyle = 'rgba(0,0,0,.45)'; c.fillRect(x - 40, y - 52, 80, 40); c.restore();
    c.strokeRect(x - 40, y - 52, 80, 40);
    c.font = `700 32px ${MONO}`; c.textAlign = 'center';
    c.fillText(pad(Math.round(s.hdg) % 360 || 360, 3), x, y - 31);
    c.font = `700 36px ${MONO}`;
  }

  // ---- the bank scale: ticks fixed to the glass, a pointer for the sky ----
  {
    const [x, y] = at(0, -3.2);
    const R = 238;
    for (const a of [-45, -30, -20, -10, 0, 10, 20, 30, 45]) {
      const sn = Math.sin(a * RAD), cs = Math.cos(a * RAD), L = a % 30 === 0 ? 18 : 10;
      c.beginPath(); c.moveTo(x + R * sn, y + R * cs); c.lineTo(x + (R + L) * sn, y + (R + L) * cs); c.stroke();
    }
    const a = clamp(-s.roll, -55, 55) * RAD;
    const px = x + (R - 4) * Math.sin(a), py = y + (R - 4) * Math.cos(a);
    c.save(); c.translate(px, py); c.rotate(-a);
    c.beginPath(); c.moveTo(0, 0); c.lineTo(-9, -17); c.lineTo(9, -17); c.closePath(); c.stroke(); c.restore();
  }

  // ---- the numbers ----
  c.textAlign = 'left';
  { const [x, y] = at(-8.6, 3.0); c.fillText(s.nz.toFixed(1), x, y); }
  {
    const [x, y] = at(-9.0, -7.6);
    c.font = `700 31px ${MONO}`;
    c.fillText(s.mach.toFixed(2), x, y);
    c.fillText(s.maxG.toFixed(1), x, y + 38);
    c.fillText(s.ap ? 'A/P' : s.gear > 0.5 ? 'LDG' : 'NAV', x, y + 76);
    if (s.sb > 0.1) c.fillText('SB', x + 86, y + 76);
  }
  if (s.stpt) {
    const [x, y] = at(3.9, -10.4);
    c.font = `700 31px ${MONO}`; c.textAlign = 'left';
    c.fillText(`${pad(Math.min(s.stpt.nm, 99.9) * 10, 3).replace(/(\d)$/, '.$1')}>${pad(s.stpt.n, 2)}`, x, y);
  }

  // ---- what the aircraft is trying to tell you ----
  c.textAlign = 'center';
  if (s.pull) {
    // The break X: the ground is closer than the pull-out.
    if (blink) {
      const [x, y] = at(0, -4);
      c.save(); c.lineWidth = 9;
      c.beginPath(); c.moveTo(x - 190, y - 190); c.lineTo(x + 190, y + 190); c.moveTo(x + 190, y - 190); c.lineTo(x - 190, y + 190); c.stroke();
      c.restore();
    }
  } else if (blink) {
    const [x, y] = at(0, -6.8);
    c.font = `800 44px ${MONO}`;
    const w = s.over ? 'WARN' : s.limit ? 'LIMIT' : s.fuelLow ? 'FUEL' : s.config ? 'WARN' : '';
    if (w) c.fillText(w, x, y);
  }
  c.restore();
  return fpm;
}

/* ====================================================================
 * The panel
 * ==================================================================== */

/* The panel's face in metres: x from its centreline, y down from its top
 * edge. f16_pit.js builds the plane from PANEL and hangs each live part in
 * front of it at its place here. */
export const PANEL = { W: 0.88, H: 0.60, MAIN_H: 0.335, PED_W: 0.25, PPM: 2300 };

/* The atlas: one canvas, every live part a rectangle of it. */
export const ATLAS = {
  N: 1024,
  mfdL: [0, 0, 320, 320], mfdR: [320, 0, 320, 320],
  adi: [640, 0, 192, 192], hsi: [832, 0, 192, 192],
  asi: [640, 192, 128, 128], altm: [768, 192, 128, 128], fuel: [896, 192, 128, 128],
  ded: [0, 320, 448, 128],
  oil: [448, 320, 128, 128], noz: [576, 320, 128, 128], rpm: [704, 320, 128, 128], ftit: [832, 320, 128, 128],
  aoa: [0, 448, 64, 176], vvi: [64, 448, 64, 176],
  rwr: [128, 448, 176, 176],
  aoaIdx: [304, 448, 48, 120], nwsIdx: [352, 448, 48, 120],
  browL: [400, 448, 160, 72], browR: [560, 448, 320, 72],
  gearL: [400, 520, 160, 80], ffl: [560, 520, 160, 64], sbi: [720, 520, 96, 64], clock: [816, 520, 128, 64]
};

/* Where each live part sits on the panel: centre x, centre y, width, height,
 * metres. The upfront parts (rwr, ded, the indexers, the eyebrows) are not
 * on the panel's plane; f16_pit.js places those. */
export const PARTS = {
  mfdL: [-0.158, 0.172, 0.104, 0.104], mfdR: [0.158, 0.172, 0.104, 0.104],
  asi: [-0.046, 0.045, 0.062, 0.062], altm: [0.046, 0.045, 0.062, 0.062],
  adi: [0, 0.128, 0.086, 0.086],
  aoa: [-0.066, 0.128, 0.022, 0.078], vvi: [0.066, 0.128, 0.022, 0.078],
  hsi: [0, 0.238, 0.090, 0.090],
  oil: [0.282, 0.050, 0.054, 0.054], noz: [0.282, 0.112, 0.054, 0.054],
  rpm: [0.282, 0.174, 0.054, 0.054], ftit: [0.282, 0.236, 0.054, 0.054],
  fuel: [0.365, 0.105, 0.062, 0.062], ffl: [0.365, 0.170, 0.060, 0.024], clock: [0.365, 0.215, 0.052, 0.026],
  gearL: [-0.325, 0.215, 0.076, 0.038], sbi: [-0.386, 0.275, 0.034, 0.022]
};

const rr = (c, x, y, w, h, r) => {
  c.beginPath(); c.moveTo(x + r, y); c.lineTo(x + w - r, y); c.quadraticCurveTo(x + w, y, x + w, y + r);
  c.lineTo(x + w, y + h - r); c.quadraticCurveTo(x + w, y + h, x + w - r, y + h); c.lineTo(x + r, y + h);
  c.quadraticCurveTo(x, y + h, x, y + h - r); c.lineTo(x, y + r); c.quadraticCurveTo(x, y, x + r, y); c.closePath();
};

/** The panel's face, drawn once. The canvas is PANEL.W x PANEL.H metres at
 *  PANEL.PPM pixels a metre; the legs' wells either side of the pedestal are
 *  left transparent. */
export function drawPanel(c) {
  const k = PANEL.PPM, W = PANEL.W * k, H = PANEL.H * k;
  const X = (m) => (m + PANEL.W / 2) * k, Y = (m) => m * k;
  c.clearRect(0, 0, W, H);
  // The grey of an F-16's office, a shade the manuals call 36231, lit from above.
  const g = c.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, '#3a3e42'); g.addColorStop(0.5, '#4a4f54'); g.addColorStop(1, '#3b3f43');
  c.fillStyle = g;
  c.fillRect(0, 0, W, Y(PANEL.MAIN_H));
  c.fillRect(X(-PANEL.PED_W / 2), Y(PANEL.MAIN_H) - 1, PANEL.PED_W * k, H);
  // A little wear, so it is a surface and not a fill.
  for (let i = 0; i < 2600; i++) {
    c.fillStyle = `rgba(${Math.random() < 0.5 ? '0,0,0' : '255,255,255'},${(Math.random() * 0.035).toFixed(3)})`;
    c.fillRect(Math.random() * W, Math.random() * H, 1 + Math.random() * 5, 1 + Math.random() * 2);
  }
  const label = (t, x, y, px = 0.0052, col = '#e9ecef', align = 'center') => {
    c.fillStyle = col; c.font = `700 ${Math.round(px * k)}px ${MONO}`; c.textAlign = align; c.textBaseline = 'middle';
    c.fillText(t, X(x), Y(y));
  };
  const plate = (x, y, w, h, fill = '#1b1d20') => {
    rr(c, X(x - w / 2), Y(y - h / 2), w * k, h * k, 0.004 * k); c.fillStyle = fill; c.fill();
    c.strokeStyle = 'rgba(0,0,0,.6)'; c.lineWidth = 2; c.stroke();
    c.strokeStyle = 'rgba(255,255,255,.07)'; c.lineWidth = 1; c.stroke();
    for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      c.beginPath(); c.arc(X(x + sx * (w / 2 - 0.005)), Y(y + sy * (h / 2 - 0.005)), 0.0018 * k, 0, 7);
      c.fillStyle = '#0c0d0e'; c.fill();
    }
  };
  const round = (x, y, d) => {
    c.beginPath(); c.arc(X(x), Y(y), (d / 2 + 0.004) * k, 0, 7); c.fillStyle = '#131416'; c.fill();
    c.strokeStyle = '#5b6066'; c.lineWidth = 2; c.stroke();
  };
  const sw = (x, y, on) => {            // a toggle switch
    c.beginPath(); c.arc(X(x), Y(y), 0.0046 * k, 0, 7); c.fillStyle = '#0e0f10'; c.fill();
    c.strokeStyle = '#c9ced4'; c.lineWidth = 0.0022 * k; c.lineCap = 'round';
    c.beginPath(); c.moveTo(X(x), Y(y)); c.lineTo(X(x), Y(y + (on ? -0.008 : 0.008))); c.stroke();
  };
  const knob = (x, y, r = 0.008) => {
    c.beginPath(); c.arc(X(x), Y(y), r * k, 0, 7); c.fillStyle = '#1a1b1d'; c.fill();
    c.strokeStyle = '#6a7077'; c.lineWidth = 2; c.stroke();
    c.strokeStyle = '#e9ecef'; c.lineWidth = 3; c.beginPath(); c.moveTo(X(x), Y(y)); c.lineTo(X(x + r * 0.6), Y(y - r * 0.7)); c.stroke();
  };

  // ---- the two MFDs: a bezel, twenty buttons, a rocker at each corner ----
  for (const sx of [-1, 1]) {
    const [cx, cy, w] = PARTS[sx < 0 ? 'mfdL' : 'mfdR'];
    const bw = w + 0.046;
    rr(c, X(cx - bw / 2), Y(cy - bw / 2), bw * k, bw * k, 0.008 * k);
    c.fillStyle = '#17181a'; c.fill(); c.strokeStyle = '#000'; c.lineWidth = 3; c.stroke();
    c.strokeStyle = 'rgba(255,255,255,.10)'; c.lineWidth = 1.5; c.stroke();
    c.fillStyle = '#020403'; c.fillRect(X(cx - w / 2 - 0.002), Y(cy - w / 2 - 0.002), (w + 0.004) * k, (w + 0.004) * k);
    for (let i = 0; i < 5; i++) {
      const o = (i - 2) * w / 5.2;
      for (const [bx, by] of [[cx + o, cy - w / 2 - 0.0125], [cx + o, cy + w / 2 + 0.0125], [cx - w / 2 - 0.0125, cy + o], [cx + w / 2 + 0.0125, cy + o]]) {
        rr(c, X(bx - 0.0058), Y(by - 0.0058), 0.0116 * k, 0.0116 * k, 3); c.fillStyle = '#2c2f33'; c.fill();
        c.strokeStyle = '#08090a'; c.lineWidth = 1.5; c.stroke();
        c.fillStyle = 'rgba(255,255,255,.14)'; c.fillRect(X(bx - 0.004), Y(by - 0.0045), 0.008 * k, 0.002 * k);
      }
    }
    for (const [dx, dy, t] of [[-1, -1, 'GAIN'], [1, -1, 'SYM'], [-1, 1, 'BRT'], [1, 1, 'CON']]) {
      rr(c, X(cx + dx * (bw / 2 - 0.0085) - 0.0055), Y(cy + dy * (bw / 2 - 0.0085) - 0.0055), 0.011 * k, 0.011 * k, 3);
      c.fillStyle = '#26282b'; c.fill();
      label(t, cx + dx * (bw / 2 - 0.0085), cy + dy * (bw / 2 - 0.0085), 0.0032, '#aeb4ba');
    }
  }

  // ---- the middle: the flight instruments ----
  plate(0, 0.045, 0.165, 0.078);
  round(PARTS.asi[0], PARTS.asi[1], PARTS.asi[2]); round(PARTS.altm[0], PARTS.altm[1], PARTS.altm[2]);
  plate(0, 0.128, 0.165, 0.094);
  round(0, 0.128, PARTS.adi[2]);
  for (const key of ['aoa', 'vvi']) {
    const [x, y, w, h] = PARTS[key];
    c.fillStyle = '#0b0c0d'; c.fillRect(X(x - w / 2 - 0.002), Y(y - h / 2 - 0.002), (w + 0.004) * k, (h + 0.004) * k);
  }
  label('AOA', PARTS.aoa[0], 0.172, 0.0036); label('VVI', PARTS.vvi[0], 0.172, 0.0036);
  plate(0, 0.238, 0.165, 0.106);
  round(0, 0.238, PARTS.hsi[2]);
  knob(-0.066, 0.278, 0.007); knob(0.066, 0.278, 0.007);
  label('HDG', -0.066, 0.262, 0.0032); label('CRS', 0.066, 0.262, 0.0032);

  // ---- the pedestal, below: fuel selector, the clock's old place, covers ----
  plate(0, 0.33, 0.21, 0.07);
  label('INSTR MODE', -0.06, 0.31, 0.0036); knob(-0.06, 0.335, 0.011);
  label('FUEL QTY SEL', 0.045, 0.31, 0.0036); knob(0.045, 0.335, 0.011);
  for (const [t, a] of [['TEST', -120], ['NORM', -60], ['RSVR', 0], ['INT WING', 55], ['EXT WING', 110]]) {
    label(t, 0.045 + 0.022 * Math.sin(a * RAD), 0.335 - 0.019 * Math.cos(a * RAD) + 0.004, 0.0024, '#c9ced4');
  }
  plate(0, 0.43, 0.21, 0.11);
  label('PEDAL ADJ', -0.06, 0.39, 0.0034);
  rr(c, X(-0.075), Y(0.40), 0.03 * k, 0.05 * k, 6); c.fillStyle = '#101112'; c.fill();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) {
    rr(c, X(0.0 + j * 0.022), Y(0.395 + i * 0.026), 0.019 * k, 0.021 * k, 4);
    c.fillStyle = '#0d0e0f'; c.fill(); c.strokeStyle = '#2a2c2f'; c.lineWidth = 1.5; c.stroke();
  }
  plate(0, 0.54, 0.21, 0.09, '#232629');
  label('RUDDER', -0.05, 0.515, 0.0034); label('ANTI-SKID', 0.05, 0.515, 0.0034);
  sw(-0.05, 0.545, false); sw(0.05, 0.545, true);

  // ---- right: the engine column, the fuel, the lights ----
  plate(0.282, 0.143, 0.068, 0.262);
  for (const key of ['oil', 'noz', 'rpm', 'ftit']) round(PARTS[key][0], PARTS[key][1], PARTS[key][2]);
  plate(0.368, 0.143, 0.092, 0.262);
  round(PARTS.fuel[0], PARTS.fuel[1], PARTS.fuel[2]);
  label('STBY', 0.365, 0.035, 0.0036);
  // The standby compass, a little window with a card in it.
  rr(c, X(0.343), Y(0.043), 0.044 * k, 0.018 * k, 4); c.fillStyle = '#0a0b0c'; c.fill();
  label('FUEL FLOW', 0.365, 0.152, 0.0032); label('PPH', 0.365, 0.188, 0.0028, '#aeb4ba');
  // The caution panel: a grid of legends, dark until something is wrong.
  const cau = ['FLCS FAULT', 'ENGINE FAULT', 'AVIONICS', 'SEAT NOT ARMED', 'ELEC SYS', 'SEC', 'EQUIP HOT', 'NWS FAIL', 'PROBE HEAT', 'FUEL/OIL HOT', 'RADAR ALT', 'ANTI SKID'];
  cau.forEach((t, i) => {
    const x = 0.33 + (i % 3) * 0.0295, y = 0.245 + Math.floor(i / 3) * 0.0125;
    c.fillStyle = '#16130f'; c.fillRect(X(x - 0.014), Y(y - 0.0055), 0.028 * k, 0.011 * k);
    c.strokeStyle = '#000'; c.lineWidth = 1; c.strokeRect(X(x - 0.014), Y(y - 0.0055), 0.028 * k, 0.011 * k);
    label(t, x, y, 0.0021, 'rgba(255,190,90,.28)');
  });
  plate(0.33, 0.31, 0.17, 0.035, '#2a2d30');
  label('AIR SOURCE', 0.29, 0.30, 0.0032); knob(0.29, 0.316, 0.006);
  label('TEMP', 0.36, 0.30, 0.0032); knob(0.36, 0.316, 0.006);

  // ---- left: the misc panel, the gear ----
  plate(-0.325, 0.07, 0.165, 0.115);
  label('MISC', -0.385, 0.024, 0.0038);
  label('MASTER ARM', -0.36, 0.045, 0.0032); sw(-0.36, 0.068, false);
  label('LASER', -0.305, 0.045, 0.0032); sw(-0.305, 0.068, false);
  label('ROLL', -0.375, 0.096, 0.0032); label('PITCH', -0.33, 0.096, 0.0032); label('RF', -0.285, 0.096, 0.0032);
  label('HDG SEL', -0.375, 0.102, 0.0022, '#aeb4ba'); label('ALT HOLD', -0.33, 0.102, 0.0022, '#aeb4ba');
  sw(-0.375, 0.115, true); sw(-0.33, 0.115, true); sw(-0.285, 0.115, true);
  label('AUTOPILOT', -0.352, 0.087, 0.0028, '#aeb4ba');
  plate(-0.325, 0.215, 0.165, 0.155);
  label('LANDING GEAR', -0.325, 0.148, 0.0038);
  label('WHEELS DOWN', PARTS.gearL[0], PARTS.gearL[1] - 0.027, 0.003);
  c.fillStyle = '#0b0c0d'; c.fillRect(X(PARTS.gearL[0] - 0.04), Y(PARTS.gearL[1] - 0.021), 0.08 * k, 0.042 * k);
  // The handle's slot; the handle itself is built, and moves.
  rr(c, X(-0.262), Y(0.17), 0.016 * k, 0.085 * k, 6); c.fillStyle = '#08090a'; c.fill();
  label('UP', -0.254, 0.162, 0.003); label('DN', -0.254, 0.262, 0.003);
  label('HOOK', -0.385, 0.165, 0.003); sw(-0.385, 0.184, true);
  label('LIGHTS', -0.385, 0.215, 0.003); sw(-0.385, 0.234, false);
  label('SPD BRK', -0.386, 0.258, 0.003);
  c.fillStyle = '#0b0c0d'; c.fillRect(X(PARTS.sbi[0] - 0.019), Y(PARTS.sbi[1] - 0.013), 0.038 * k, 0.026 * k);
  plate(-0.325, 0.314, 0.165, 0.03, '#2a2d30');
  label('CMDS', -0.385, 0.314, 0.0034);
  for (let i = 0; i < 4; i++) sw(-0.35 + i * 0.028, 0.316, i < 2);
}

/* ---------- the live parts ---------- */

const begin = (c, key) => {
  const [x, y, w, h] = ATLAS[key];
  c.save(); c.beginPath(); c.rect(x, y, w, h); c.clip(); c.translate(x, y);
  c.clearRect(0, 0, w, h);
  return [w, h];
};

function dial(c, key, s, o) {
  const [w] = begin(c, key);
  const R = w / 2, cx = R, cy = R;
  c.fillStyle = '#0c0d0e'; c.beginPath(); c.arc(cx, cy, R - 1, 0, 7); c.fill();
  c.strokeStyle = '#e8ebee'; c.fillStyle = '#e8ebee'; c.lineCap = 'butt';
  c.font = `700 ${Math.round(w * 0.105)}px ${MONO}`; c.textAlign = 'center'; c.textBaseline = 'middle';
  const A = (v) => (o.a0 + (o.a1 - o.a0) * (v - o.min) / (o.max - o.min)) * RAD;
  for (let v = o.min; v <= o.max + 1e-6; v += o.minor) {
    const a = A(v), big = Math.abs(v / o.major - Math.round(v / o.major)) < 1e-6;
    c.lineWidth = big ? 2.4 : 1.3;
    c.beginPath(); c.moveTo(cx + Math.sin(a) * (R - 5), cy - Math.cos(a) * (R - 5));
    c.lineTo(cx + Math.sin(a) * (R - (big ? 16 : 11)), cy - Math.cos(a) * (R - (big ? 16 : 11))); c.stroke();
    if (big && (!o.skip || !o.skip(v))) c.fillText(o.fmt ? o.fmt(v) : String(v), cx + Math.sin(a) * (R - 27), cy - Math.cos(a) * (R - 27));
  }
  if (o.red) {
    c.strokeStyle = '#e5483a'; c.lineWidth = 4;
    c.beginPath(); c.arc(cx, cy, R - 4, A(o.red[0]) - Math.PI / 2, A(o.red[1]) - Math.PI / 2); c.stroke();
  }
  c.font = `700 ${Math.round(w * 0.085)}px ${MONO}`; c.fillStyle = '#c9ced4';
  c.fillText(o.name, cx, cy + R * 0.34);
  if (o.sub) c.fillText(o.sub, cx, cy + R * 0.52);
  const a = A(clamp(s, o.min, o.max));
  c.strokeStyle = '#fff'; c.lineWidth = 3.4; c.lineCap = 'round';
  c.beginPath(); c.moveTo(cx - Math.sin(a) * 9, cy + Math.cos(a) * 9); c.lineTo(cx + Math.sin(a) * (R - 13), cy - Math.cos(a) * (R - 13)); c.stroke();
  c.fillStyle = '#d7dadd'; c.beginPath(); c.arc(cx, cy, 5.5, 0, 7); c.fill();
  c.restore();
}

/** Airspeed: a dial that is fine where the approach is flown and coarse
 *  where nobody reads it, and the Mach number in a window. */
const ASI = [[0, 0], [100, 40], [150, 80], [200, 122], [300, 190], [400, 238], [500, 272], [600, 298], [700, 318], [800, 336]];
function asiAngle(kt) {
  kt = clamp(kt, 0, 800);
  for (let i = 1; i < ASI.length; i++) if (kt <= ASI[i][0]) {
    const [k0, a0] = ASI[i - 1], [k1, a1] = ASI[i];
    return (a0 + (a1 - a0) * (kt - k0) / (k1 - k0)) * RAD;
  }
  return 336 * RAD;
}
function drawAsi(c, s) {
  const [w] = begin(c, 'asi');
  const R = w / 2;
  c.fillStyle = '#0c0d0e'; c.beginPath(); c.arc(R, R, R - 1, 0, 7); c.fill();
  c.strokeStyle = '#e8ebee'; c.fillStyle = '#e8ebee'; c.textAlign = 'center'; c.textBaseline = 'middle';
  c.font = `700 12px ${MONO}`;
  for (let v = 60; v <= 800; v += v < 200 ? 10 : v < 400 ? 20 : 50) {
    const a = asiAngle(v), big = v % 100 === 0 || (v < 200 && v % 50 === 0);
    c.lineWidth = big ? 2.2 : 1.1;
    c.beginPath(); c.moveTo(R + Math.sin(a) * (R - 4), R - Math.cos(a) * (R - 4));
    c.lineTo(R + Math.sin(a) * (R - (big ? 14 : 9)), R - Math.cos(a) * (R - (big ? 14 : 9))); c.stroke();
    if (v % 100 === 0) c.fillText(String(v / 100), R + Math.sin(a) * (R - 23), R - Math.cos(a) * (R - 23));
  }
  c.fillStyle = '#000'; c.fillRect(R - 22, R + 14, 44, 20); c.strokeStyle = '#6a7077'; c.lineWidth = 1; c.strokeRect(R - 22, R + 14, 44, 20);
  c.fillStyle = '#fff'; c.font = `700 14px ${MONO}`; c.fillText(s.mach.toFixed(2), R, R + 25);
  c.fillStyle = '#c9ced4'; c.font = `700 9px ${MONO}`; c.fillText('KNOTS x100', R, R - 20); c.fillText('MACH', R, R + 43);
  const a = asiAngle(s.cas);
  c.strokeStyle = '#fff'; c.lineWidth = 3.4; c.lineCap = 'round';
  c.beginPath(); c.moveTo(R, R); c.lineTo(R + Math.sin(a) * (R - 12), R - Math.cos(a) * (R - 12)); c.stroke();
  c.fillStyle = '#d7dadd'; c.beginPath(); c.arc(R, R, 5, 0, 7); c.fill();
  c.restore();
}

function drawAlt(c, s) {
  const [w] = begin(c, 'altm');
  const R = w / 2;
  c.fillStyle = '#0c0d0e'; c.beginPath(); c.arc(R, R, R - 1, 0, 7); c.fill();
  c.strokeStyle = '#e8ebee'; c.fillStyle = '#e8ebee'; c.textAlign = 'center'; c.textBaseline = 'middle';
  c.font = `700 14px ${MONO}`;
  for (let i = 0; i < 50; i++) {
    const a = i / 50 * 2 * Math.PI, big = i % 5 === 0;
    c.lineWidth = big ? 2.2 : 1.1;
    c.beginPath(); c.moveTo(R + Math.sin(a) * (R - 4), R - Math.cos(a) * (R - 4));
    c.lineTo(R + Math.sin(a) * (R - (big ? 14 : 9)), R - Math.cos(a) * (R - (big ? 14 : 9))); c.stroke();
    if (big) c.fillText(String(i / 5), R + Math.sin(a) * (R - 24), R - Math.cos(a) * (R - 24));
  }
  // The drum: ten-thousands, thousands, hundreds.
  const ft = Math.max(0, s.alt);
  c.fillStyle = '#000'; c.fillRect(R - 32, R - 12, 64, 22); c.strokeStyle = '#6a7077'; c.lineWidth = 1; c.strokeRect(R - 32, R - 12, 64, 22);
  c.fillStyle = '#fff'; c.font = `700 16px ${MONO}`; c.fillText(pad(Math.floor(ft / 100), 3) + '00', R, R);
  c.fillStyle = '#c9ced4'; c.font = `700 9px ${MONO}`; c.fillText('ALT', R, R - 24); c.fillText('29.92', R, R + 24);
  const a = (ft % 1000) / 1000 * 2 * Math.PI;
  c.strokeStyle = '#fff'; c.lineWidth = 3; c.lineCap = 'round';
  c.beginPath(); c.moveTo(R + Math.sin(a) * 30, R - Math.cos(a) * 30); c.lineTo(R + Math.sin(a) * (R - 10), R - Math.cos(a) * (R - 10)); c.stroke();
  c.restore();
}

/** The attitude indicator: the ball, the aircraft, the bank scale, the ball
 *  in the tube. */
function drawAdi(c, s) {
  const [w] = begin(c, 'adi');
  const R = w / 2, k = R / 34;                       // pixels per degree of pitch
  c.beginPath(); c.arc(R, R, R - 2, 0, 7); c.clip();
  c.save();
  c.translate(R, R); c.rotate(-s.roll * RAD); c.translate(0, s.pitch * k);
  c.fillStyle = '#5d8fc4'; c.fillRect(-2 * w, -3 * w, 4 * w, 3 * w);
  c.fillStyle = '#4b3a2a'; c.fillRect(-2 * w, 0, 4 * w, 3 * w);
  c.strokeStyle = '#fff'; c.fillStyle = '#fff'; c.lineWidth = 2;
  c.beginPath(); c.moveTo(-2 * w, 0); c.lineTo(2 * w, 0); c.stroke();
  c.font = `700 11px ${MONO}`; c.textAlign = 'center'; c.textBaseline = 'middle';
  for (let p = -90; p <= 90; p += 5) {
    if (!p || Math.abs(p - s.pitch) > 32) continue;
    const y = -p * k, L = p % 10 === 0 ? 22 : 10;
    c.lineWidth = 1.5; c.beginPath(); c.moveTo(-L, y); c.lineTo(L, y); c.stroke();
    if (p % 10 === 0) { c.fillText(String(Math.abs(p)), -L - 11, y); c.fillText(String(Math.abs(p)), L + 11, y); }
  }
  c.restore();
  // The bank scale at the top, and its pointer.
  c.strokeStyle = '#fff'; c.lineWidth = 1.6;
  for (const a of [-60, -30, -20, -10, 0, 10, 20, 30, 60]) {
    const sn = Math.sin(a * RAD), cs = Math.cos(a * RAD), L = a % 30 === 0 ? 11 : 6;
    c.beginPath(); c.moveTo(R + sn * (R - 3), R - cs * (R - 3)); c.lineTo(R + sn * (R - 3 - L), R - cs * (R - 3 - L)); c.stroke();
  }
  c.save(); c.translate(R, R); c.rotate(-s.roll * RAD);
  c.fillStyle = '#fff'; c.beginPath(); c.moveTo(0, -R + 15); c.lineTo(-6, -R + 26); c.lineTo(6, -R + 26); c.closePath(); c.fill(); c.restore();
  // The aircraft: fixed, orange, a W.
  c.strokeStyle = '#ffb000'; c.lineWidth = 4; c.lineCap = 'round'; c.lineJoin = 'round';
  c.beginPath(); c.moveTo(R - 46, R); c.lineTo(R - 18, R); c.lineTo(R - 9, R + 9); c.lineTo(R, R); c.lineTo(R + 9, R + 9); c.lineTo(R + 18, R); c.lineTo(R + 46, R); c.stroke();
  // Slip: the ball goes to the outside of a skid.
  c.fillStyle = '#0c0d0e'; c.fillRect(R - 32, w - 24, 64, 12);
  c.fillStyle = '#fff'; c.beginPath(); c.arc(R + clamp(s.beta * 6, -26, 26), w - 18, 5, 0, 7); c.fill();
  c.strokeStyle = '#fff'; c.lineWidth = 1; c.strokeRect(R - 7, w - 24, 14, 12);
  c.restore();
}

/** The horizontal situation indicator: the card, the lubber line, a course
 *  arrow to the steerpoint, miles in a window. */
function drawHsi(c, s) {
  const [w] = begin(c, 'hsi');
  const R = w / 2;
  c.fillStyle = '#0c0d0e'; c.beginPath(); c.arc(R, R, R - 1, 0, 7); c.fill();
  c.save(); c.translate(R, R); c.rotate(-s.hdg * RAD);
  c.strokeStyle = '#e8ebee'; c.fillStyle = '#e8ebee'; c.textAlign = 'center'; c.textBaseline = 'middle';
  for (let a = 0; a < 360; a += 5) {
    c.save(); c.rotate(a * RAD);
    const big = a % 10 === 0;
    c.lineWidth = big ? 2 : 1; c.beginPath(); c.moveTo(0, -R + 4); c.lineTo(0, -R + (big ? 15 : 10)); c.stroke();
    if (a % 30 === 0) {
      c.font = `700 ${a % 90 === 0 ? 17 : 13}px ${MONO}`;
      c.fillText(a === 0 ? 'N' : a === 90 ? 'E' : a === 180 ? 'S' : a === 270 ? 'W' : String(a / 10), 0, -R + 27);
    }
    c.restore();
  }
  if (s.stpt) {
    c.rotate(s.stpt.brg * RAD);
    c.strokeStyle = '#fff'; c.fillStyle = '#fff'; c.lineWidth = 3;
    c.beginPath(); c.moveTo(0, R - 42); c.lineTo(0, -R + 46); c.stroke();
    c.beginPath(); c.moveTo(0, -R + 36); c.lineTo(-8, -R + 50); c.lineTo(8, -R + 50); c.closePath(); c.fill();
  }
  c.restore();
  c.strokeStyle = '#ffb000'; c.lineWidth = 3;
  c.beginPath(); c.moveTo(R, 3); c.lineTo(R, 17); c.stroke();
  // The aircraft, in the middle, nose up the card.
  c.strokeStyle = '#ffb000'; c.lineWidth = 2.5;
  c.beginPath(); c.moveTo(R, R - 13); c.lineTo(R, R + 12); c.moveTo(R - 12, R); c.lineTo(R + 12, R); c.moveTo(R - 5, R + 10); c.lineTo(R + 5, R + 10); c.stroke();
  if (s.stpt) {
    c.fillStyle = '#000'; c.fillRect(4, 4, 46, 19); c.fillRect(w - 50, 4, 46, 19);
    c.fillStyle = '#fff'; c.font = `700 13px ${MONO}`; c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(pad(Math.min(999, s.stpt.nm * 10), 3).replace(/(\d)$/, '.$1'), 27, 14);
    c.fillText(pad(((s.stpt.brg % 360) + 360) % 360, 3), w - 27, 14);
  }
  c.restore();
}

function drawTape(c, key, val, lo, hi, step, major, fmt, bands) {
  const [w, h] = begin(c, key);
  c.fillStyle = '#0c0d0e'; c.fillRect(0, 0, w, h);
  const span = (hi - lo), k = h / span * 2.2;       // a window on a longer tape
  if (bands) for (const [a, b, col] of bands) {
    c.fillStyle = col; c.fillRect(w - 9, h / 2 - (b - val) * k, 9, (b - a) * k);
  }
  c.strokeStyle = '#e8ebee'; c.fillStyle = '#e8ebee'; c.font = `700 13px ${MONO}`; c.textAlign = 'right'; c.textBaseline = 'middle';
  for (let v = lo; v <= hi + 1e-6; v += step) {
    const y = h / 2 - (v - val) * k;
    if (y < -8 || y > h + 8) continue;
    const big = Math.abs(v / major - Math.round(v / major)) < 1e-6;
    c.lineWidth = big ? 2 : 1; c.beginPath(); c.moveTo(w - 10, y); c.lineTo(w - (big ? 24 : 17), y); c.stroke();
    if (big) c.fillText(fmt(v), w - 27, y);
  }
  c.strokeStyle = '#ffb000'; c.lineWidth = 3; c.beginPath(); c.moveTo(0, h / 2); c.lineTo(w, h / 2); c.stroke();
  c.restore();
}

const SCOPE = '#7dff8e';
const osb = (c, w, h, top, bottom, left, right, hot) => {
  c.font = `700 15px ${MONO}`; c.textBaseline = 'middle';
  const put = (t, x, y, al) => {
    if (!t) return;
    c.textAlign = al;
    if (hot && hot.includes(t)) {
      const tw = c.measureText(t).width + 8;
      c.fillStyle = SCOPE; c.fillRect(al === 'center' ? x - tw / 2 : al === 'left' ? x - 4 : x - tw + 4, y - 10, tw, 20);
      c.fillStyle = '#021'; c.fillText(t, x, y); c.fillStyle = SCOPE;
    } else c.fillText(t, x, y);
  };
  top.forEach((t, i) => put(t, w * (0.14 + i * 0.18), 13, 'center'));
  bottom.forEach((t, i) => put(t, w * (0.14 + i * 0.18), h - 12, 'center'));
  left.forEach((t, i) => put(t, 6, h * (0.2 + i * 0.15), 'left'));
  right.forEach((t, i) => put(t, w - 6, h * (0.2 + i * 0.15), 'right'));
};

/** The left MFD: the radar, in range-while-search, with nothing to find. */
function drawFcr(c, s) {
  const [w, h] = begin(c, 'mfdL');
  c.fillStyle = '#010502'; c.fillRect(0, 0, w, h);
  c.strokeStyle = SCOPE; c.fillStyle = SCOPE; c.lineWidth = 1.5;
  osb(c, w, h, ['CRM', 'RWS', 'NORM', 'OVRD', 'CNTL'], ['SWAP', 'FCR', '', '', 'DCLT'], ['', 'A', '6', '', ''], ['', '', '', '', ''], ['FCR']);
  c.font = `700 17px ${MONO}`; c.textAlign = 'left'; c.fillText('40', 8, h * 0.2);
  c.fillText('▲', 9, h * 0.1 + 20); c.fillText('▼', 9, h * 0.3 + 8);
  c.textAlign = 'right'; c.fillText('B', w - 8, h * 0.35); c.fillText('4', w - 8, h * 0.5);
  // Range and azimuth ticks.
  const x0 = 44, x1 = w - 30, y0 = 36, y1 = h - 44;
  for (let i = 1; i < 4; i++) { const y = y0 + (y1 - y0) * i / 4; c.beginPath(); c.moveTo(x1 - 8, y); c.lineTo(x1, y); c.stroke(); }
  for (let i = -3; i <= 3; i++) { const x = (x0 + x1) / 2 + i * (x1 - x0) / 7; c.beginPath(); c.moveTo(x, y1); c.lineTo(x, y1 - (i === 0 ? 12 : 7)); c.stroke(); }
  // The horizon line: the radar's own, so the scope says which way is up.
  c.save(); c.translate((x0 + x1) / 2, (y0 + y1) / 2 + clamp(s.pitch, -30, 30) * 1.6); c.rotate(-s.roll * RAD);
  c.beginPath(); c.moveTo(-58, 0); c.lineTo(-16, 0); c.moveTo(16, 0); c.lineTo(58, 0); c.moveTo(-58, 0); c.lineTo(-58, 8); c.moveTo(58, 0); c.lineTo(58, 8); c.stroke();
  c.restore();
  // The acquisition cursor, and the antenna sweeping its sixty degrees.
  const cxm = (x0 + x1) / 2, cym = y0 + (y1 - y0) * 0.42;
  c.beginPath(); c.moveTo(cxm - 7, cym - 10); c.lineTo(cxm - 7, cym + 10); c.moveTo(cxm + 7, cym - 10); c.lineTo(cxm + 7, cym + 10); c.stroke();
  const ph = (s.t * 0.45) % 2, sw = ph < 1 ? ph : 2 - ph;
  const ax = x0 + 10 + (x1 - x0 - 20) * sw;
  c.beginPath(); c.moveTo(ax, y1 + 14); c.lineTo(ax - 6, y1 + 24); c.lineTo(ax + 6, y1 + 24); c.closePath(); c.stroke();
  c.fillStyle = 'rgba(125,255,142,.07)'; c.fillRect(Math.min(ax, ax - (ph < 1 ? 26 : -26)), y0, 26, y1 - y0);
  c.restore();
}

/** The right MFD: the horizontal situation display. The aircraft low on the
 *  screen, heading up, range rings, and the trails around drawn where they
 *  are: bright once found, faint before. */
function drawHsd(c, s) {
  const [w, h] = begin(c, 'mfdR');
  c.fillStyle = '#010502'; c.fillRect(0, 0, w, h);
  c.strokeStyle = SCOPE; c.fillStyle = SCOPE;
  const range = s.alt < 4000 ? 4000 : s.alt < 12000 ? 8000 : 16000;        // metres to the outer ring
  const ox = w / 2, oy = h * 0.66, ppm = (h * 0.5) / range;
  c.save(); c.beginPath(); c.rect(22, 24, w - 44, h - 48); c.clip();
  c.lineWidth = 1.3; c.strokeStyle = 'rgba(125,255,142,.5)';
  for (const r of [0.5, 1]) { c.beginPath(); c.arc(ox, oy, range * r * ppm, 0, 7); c.stroke(); }
  const b = -s.hdg * RAD, sb = Math.sin(b), cb = Math.cos(b);
  // North, on the outer ring.
  c.fillStyle = SCOPE; c.font = `700 14px ${MONO}`; c.textAlign = 'center'; c.textBaseline = 'middle';
  c.fillText('N', ox + sb * range * ppm, oy - cb * range * ppm);
  if (s.world && s.own) {
    const [px, py] = s.own;
    const P = (x, y) => { const dx = x - px, dy = y - py; return [ox + (dx * cb - dy * sb) * ppm, oy - (dx * sb + dy * cb) * ppm]; };
    for (const e of s.world) {
      const g = e.g || 0, f = e.xy[0];
      if (Math.hypot(f[0] - px, f[1] - py) > range * 2.2) continue;
      c.strokeStyle = c.fillStyle = `rgba(125,255,142,${(0.3 + 0.7 * g).toFixed(2)})`;
      c.lineWidth = g > 0.3 ? 2.2 : 1.2;
      if (e.line) {
        c.beginPath();
        for (let i = 0; i < e.xy.length; i++) { const [x, y] = P(e.xy[i][0], e.xy[i][1]); if (i) c.lineTo(x, y); else c.moveTo(x, y); }
        c.stroke();
      } else { const [x, y] = P(f[0], f[1]); c.fillRect(x - 2, y - 2, 4, 4); }
    }
    if (s.stpt && s.stpt.at) {
      const [x, y] = P(s.stpt.at[0], s.stpt.at[1]);
      c.strokeStyle = '#fff'; c.lineWidth = 2; c.beginPath(); c.arc(x, y, 7, 0, 7); c.stroke();
      c.setLineDash([5, 5]); c.beginPath(); c.moveTo(ox, oy); c.lineTo(x, y); c.stroke(); c.setLineDash([]);
    }
  }
  // Own ship.
  c.strokeStyle = '#eafff0'; c.lineWidth = 2.2;
  c.beginPath(); c.moveTo(ox, oy - 11); c.lineTo(ox, oy + 9); c.moveTo(ox - 9, oy); c.lineTo(ox + 9, oy); c.moveTo(ox - 4, oy + 8); c.lineTo(ox + 4, oy + 8); c.stroke();
  c.restore();
  c.fillStyle = SCOPE; c.strokeStyle = SCOPE;
  osb(c, w, h, ['DEP', 'DCPL', 'NORM', '', 'CNTL'], ['SWAP', '', 'HSD', '', 'DCLT'], ['', '', '', '', ''], ['', '', '', '', ''], ['HSD']);
  c.font = `700 17px ${MONO}`; c.textAlign = 'left'; c.textBaseline = 'middle';
  c.fillText(String(Math.round(range / 1852)), 8, h * 0.2);
  c.fillText('▲', 9, h * 0.1 + 20); c.fillText('▼', 9, h * 0.3 + 8);
  c.restore();
}

/** The data entry display: the steerpoint, which here is the nearest trail,
 *  by name. */
function drawDed(c, s) {
  const [w, h] = begin(c, 'ded');
  c.fillStyle = '#020a04'; c.fillRect(0, 0, w, h);
  c.fillStyle = '#9dffa8'; c.textBaseline = 'middle'; c.direction = 'ltr';
  c.font = `700 22px ${MONO}`; c.textAlign = 'left';
  const d = new Date();
  if (s.stpt) {
    c.fillText(`STPT ${pad(s.stpt.n, 2)} ↕`, 12, 20);
    c.textAlign = 'right'; c.fillText(`${pad(((s.stpt.brg % 360) + 360) % 360, 3)}°/${s.stpt.nm.toFixed(1)}NM`, w - 12, 20);
    c.save(); c.direction = 'rtl'; c.textAlign = 'center'; c.font = '700 30px system-ui, "DejaVu Sans", sans-serif';
    let name = s.stpt.name || '';
    while (name.length > 4 && c.measureText(name).width > w - 30) name = name.slice(0, -2);
    if (name !== (s.stpt.name || '')) name += '…';
    c.fillText(name, w / 2, 62); c.restore();
  } else {
    c.fillText('STPT  --', 12, 20);
    c.textAlign = 'center'; c.fillText('NO STEERPOINT', w / 2, 62);
  }
  c.font = `700 19px ${MONO}`; c.textAlign = 'left';
  c.fillText(`UHF 225.00`, 12, 106);
  c.textAlign = 'right'; c.fillText(`${pad(d.getHours(), 2)}:${pad(d.getMinutes(), 2)}:${pad(d.getSeconds(), 2)}`, w - 12, 106);
  c.restore();
}

function drawRwr(c) {
  const [w] = begin(c, 'rwr');
  const R = w / 2;
  c.fillStyle = '#020a04'; c.beginPath(); c.arc(R, R, R - 1, 0, 7); c.fill();
  c.strokeStyle = 'rgba(125,255,142,.75)'; c.lineWidth = 1.5;
  for (const r of [0.3, 0.62, 0.94]) { c.beginPath(); c.arc(R, R, (R - 4) * r, 0, 7); c.stroke(); }
  for (let a = 0; a < 360; a += 30) {
    const sn = Math.sin(a * RAD), cs = Math.cos(a * RAD), L = a % 90 === 0 ? 12 : 6;
    c.beginPath(); c.moveTo(R + sn * (R - 6), R - cs * (R - 6)); c.lineTo(R + sn * (R - 6 - L), R - cs * (R - 6 - L)); c.stroke();
  }
  c.beginPath(); c.moveTo(R - 6, R); c.lineTo(R + 6, R); c.moveTo(R, R - 6); c.lineTo(R, R + 6); c.stroke();
  c.restore();
}

function lamp(c, x, y, w, h, lines, on, col) {
  c.fillStyle = on ? col : '#191511'; rr(c, x, y, w, h, 4); c.fill();
  c.strokeStyle = '#000'; c.lineWidth = 1.5; c.stroke();
  c.fillStyle = on ? '#150c02' : 'rgba(255,200,120,.22)';
  c.font = `800 ${lines.length > 1 ? 11 : 12}px ${MONO}`; c.textAlign = 'center'; c.textBaseline = 'middle';
  lines.forEach((t, i) => c.fillText(t, x + w / 2, y + h / 2 + (i - (lines.length - 1) / 2) * 13));
}

/** The live parts. `fast` is the attitude and what the hand flies by, every
 *  call; the rest goes round in turns so no frame pays for all of it. */
export function drawDyn(c, s, turn) {
  c.direction = 'ltr';
  drawAdi(c, s); drawAsi(c, s); drawAlt(c, s);
  drawTape(c, 'aoa', s.aoa, -10, 35, 1, 5, (v) => String(v), [[11, 12.5, '#e5c13a'], [12.5, 13.5, '#3ac86a'], [13.5, 15, '#e5483a']]);
  drawTape(c, 'vvi', clamp(s.vs / 1000, -6, 6), -6, 6, 0.5, 1, (v) => String(Math.abs(v)));
  switch (turn % 4) {
    case 0: drawHsd(c, s); break;
    case 1: drawFcr(c, s); drawHsi(c, s); break;
    case 2:
      dial(c, 'rpm', s.rpm * 100, { min: 0, max: 110, minor: 5, major: 20, a0: -150, a1: 150, name: 'RPM', sub: '%', red: [103, 110] });
      dial(c, 'noz', s.noz * 100, { min: 0, max: 100, minor: 10, major: 20, a0: -140, a1: 140, name: 'NOZ', sub: 'POS' });
      dial(c, 'ftit', s.ftit / 100, { min: 2, max: 12, minor: 0.5, major: 2, a0: -150, a1: 150, name: 'FTIT', sub: 'x100', red: [9.8, 12] });
      dial(c, 'oil', s.oil, { min: 0, max: 100, minor: 10, major: 20, a0: -140, a1: 140, name: 'OIL', sub: 'PSI' });
      break;
    default: {
      drawDed(c, s); drawRwr(c);
      // Fuel: two pointers that agree, and the total in pounds.
      dial(c, 'fuel', s.fuel / 200, { min: 0, max: 42, minor: 2, major: 10, a0: -140, a1: 140, name: 'FUEL', sub: '' });
      { const [x, y, w, h] = ATLAS.fuel; c.fillStyle = '#000'; c.fillRect(x + w / 2 - 30, y + h * 0.72, 60, 18); c.fillStyle = '#fff'; c.font = `700 13px ${MONO}`; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(pad(s.fuel / 100, 2) + '00', x + w / 2, y + h * 0.72 + 9); }
      { const [w, h] = begin(c, 'ffl'); c.fillStyle = '#000'; c.fillRect(0, 0, w, h); c.fillStyle = '#fff'; c.font = `700 34px ${MONO}`; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(pad(Math.round(s.ff / 100) * 100, 5), w / 2, h / 2 + 2); c.restore(); }
      { const [w, h] = begin(c, 'clock'); const d = new Date(); c.fillStyle = '#000'; c.fillRect(0, 0, w, h); c.fillStyle = '#ffd27a'; c.font = `700 36px ${MONO}`; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(`${pad(d.getHours(), 2)}:${pad(d.getMinutes(), 2)}`, w / 2, h / 2 + 2); c.restore(); }
      // The gear: three greens when it is down and locked, the handle's red in between.
      { const [w, h] = begin(c, 'gearL'); c.fillStyle = '#0b0c0d'; c.fillRect(0, 0, w, h);
        const dn = s.gear > 0.99, mov = s.gear > 0.01 && !dn;
        for (const [x, y] of [[w / 2, 22], [w * 0.24, 58], [w * 0.76, 58]]) { c.beginPath(); c.arc(x, y, 15, 0, 7); c.fillStyle = dn ? '#39e36b' : mov ? '#5a1b14' : '#11190f'; c.fill(); c.strokeStyle = '#000'; c.lineWidth = 2; c.stroke(); }
        c.restore(); }
      { const [w, h] = begin(c, 'sbi'); c.fillStyle = '#0b0c0d'; c.fillRect(0, 0, w, h); c.fillStyle = '#e8ebee'; c.font = `700 15px ${MONO}`; c.textAlign = 'center'; c.textBaseline = 'middle';
        if (s.sb < 0.05) c.fillText('CLOSED', w / 2, h / 2); else for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { c.beginPath(); c.arc(w / 2 + (i - 1) * 22, h / 2 + (j - 1) * 16, 5, 0, 7); c.fill(); }
        c.restore(); }
      // The angle-of-attack indexer: slow, on speed, fast. Lit with the gear.
      { const [w, h] = begin(c, 'aoaIdx'); c.fillStyle = '#0b0c0d'; c.fillRect(0, 0, w, h); c.lineWidth = 5; c.lineJoin = 'round';
        const lit = s.gear > 0.5 && !s.grounded;
        const slow = lit && s.aoa > 13.6, on = lit && s.aoa > 12.2 && s.aoa < 14.4, fast = lit && s.aoa < 12.9;
        c.strokeStyle = slow ? '#ff4a3a' : '#2a1512'; c.beginPath(); c.moveTo(8, 12); c.lineTo(w / 2, 34); c.lineTo(w - 8, 12); c.stroke();
        c.strokeStyle = on ? '#4dff7a' : '#132216'; c.beginPath(); c.arc(w / 2, h / 2, 13, 0, 7); c.stroke();
        c.strokeStyle = fast ? '#ffd23a' : '#2a2412'; c.beginPath(); c.moveTo(8, h - 12); c.lineTo(w / 2, h - 34); c.lineTo(w - 8, h - 12); c.stroke();
        c.restore(); }
      { const [w, h] = begin(c, 'nwsIdx'); c.fillStyle = '#0b0c0d'; c.fillRect(0, 0, w, h); c.font = `800 13px ${MONO}`; c.textAlign = 'center'; c.textBaseline = 'middle';
        c.fillStyle = '#12233a'; c.fillText('RDY', w / 2, 20);
        c.fillStyle = s.grounded ? '#4dff7a' : '#132216'; c.fillText('AR', w / 2, 52); c.fillText('NWS', w / 2, 68);
        c.fillStyle = '#2a2412'; c.fillText('DISC', w / 2, 100); c.restore(); }
      { const [w, h] = begin(c, 'browL'); c.clearRect(0, 0, w, h);
        lamp(c, 4, 6, 74, 60, ['MASTER', 'CAUTION'], s.caution && (s.t * 2 % 1) < 0.6, '#ffb43a');
        lamp(c, 84, 6, 72, 60, ['TF-FAIL'], false, '#ff4a3a'); c.restore(); }
      { const [w, h] = begin(c, 'browR'); c.clearRect(0, 0, w, h);
        lamp(c, 4, 6, 58, 60, ['ENG', 'FIRE'], false, '#ff4a3a');
        lamp(c, 66, 6, 58, 60, ['HYD/OIL', 'PRESS'], false, '#ff4a3a');
        lamp(c, 128, 6, 58, 60, ['FLCS', 'DBU ON'], false, '#ff4a3a');
        lamp(c, 190, 6, 62, 60, ['TO/LDG', 'CONFIG'], s.config, '#ff4a3a');
        lamp(c, 256, 6, 60, 60, ['CANOPY', 'OXY LOW'], false, '#ff4a3a'); c.restore(); }
    }
  }
}
