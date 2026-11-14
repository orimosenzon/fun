// The furnishings of the fortress, modelled in code: suits of armour, stone kings, shields over
// crossed weapons, banners, tapestries, carpets, wheel chandeliers, torches and fire baskets,
// weapon racks, barrels and chests, the portcullis, stained glass, the old linden.
// K holds one mesh builder per material: plain (painted or carved, matt), metal, cloth and glass
// (textured from the atlas), fire (glowing), stone (the masonry), and the list of lights.
import { addCyl, pbox, lathe, ball, limb, slab } from './meshkit.js';
import { CELL } from './castleTex.js';

const STONE = [0.30, 0.285, 0.25], STONE2 = [0.225, 0.215, 0.19], STEEL = [0.52, 0.54, 0.58], IRON = [0.10, 0.10, 0.11], GOLD = [0.95, 0.70, 0.24];
const OAK = [0.30, 0.20, 0.12], DARK = [0.16, 0.10, 0.065], RED = [0.5, 0.06, 0.07], WAX = [0.86, 0.80, 0.62], BLACK = [0.03, 0.03, 0.035];
const FL0 = [1.0, 0.33, 0.06], FL1 = [1.0, 0.75, 0.3];             // a flame: the orange mantle and the bright heart, drawn one over the other
const yawOf = (nx, nz) => Math.atan2(nx, nz);
function both(Bs, pos, yaw, s, roll, fn) { for (const B of Bs) B.push(pos, yaw, s, roll); fn(); for (const B of Bs) B.pop(); }

// origin at the cross-guard, blade along +y
function sword(B, L, blade = STEEL, hilt = GOLD) {
  pbox(B, -0.04, 0.04, 0.01, L - 0.1, -0.009, 0.009, blade);
  B.tri([-0.04, L - 0.1, 0], [0.04, L - 0.1, 0], [0, L, 0], blade, [0, 0, 1]); B.tri([-0.04, L - 0.1, 0], [0.04, L - 0.1, 0], [0, L, 0], blade, [0, 0, -1]);
  pbox(B, -0.13, 0.13, -0.014, 0.014, -0.02, 0.02, hilt);
  limb(B, [0, -0.014, 0], [0, -0.16, 0], 0.017, 0.017, blade === STEEL ? DARK : blade, 5);
  ball(B, [0, -0.19, 0], 0.035, hilt, { n: 6, k: 3 });
}
// origin at the foot of the shaft, which runs along +y; the blade looks along +x
function halberd(Bm, Bp, L, head = STEEL, shaft = OAK) {
  limb(Bp, [0, 0, 0], [0, L - 0.08, 0], 0.016, 0.014, shaft, 5);
  slab(Bm, [[0.012, L - 0.28], [0.13, L - 0.33], [0.155, L - 0.22], [0.13, L - 0.11], [0.012, L - 0.16]], -0.007, 0.007, head);
  slab(Bm, [[-0.012, L - 0.25], [-0.105, L - 0.215], [-0.012, L - 0.18]], -0.006, 0.006, head);
  limb(Bm, [0, L - 0.12, 0], [0, L + 0.14, 0], 0.014, 0, head, 5);
}
function arm(B, sh, el, hd, col, r) { limb(B, sh, el, r * 1.15, r, col, 6, false); ball(B, el, r * 1.12, col, { n: 6, k: 3 }); limb(B, el, hd, r, r * 0.85, col, 6, false); ball(B, hd, r * 1.05, col, { n: 6, k: 3 }); }

// a full suit of plate on a low oak plinth, 1 unit tall; pose 0 leans on a sword, pose 1 holds a halberd
function armor(K, pose) {
  const M = K.metal, P = K.plain, S = STEEL, S2 = [0.38, 0.4, 0.44];
  pbox(P, -0.3, 0.3, -0.22, -0.16, -0.24, 0.24, DARK); pbox(P, -0.25, 0.25, -0.16, 0, -0.2, 0.2, DARK);
  for (const sg of [-1, 1]) {
    ball(M, [sg * 0.1, 0.035, 0.04], 0.062, S, { n: 6, k: 3, sy: 0.6, sz: 1.7 });
    limb(M, [sg * 0.1, 0.05, 0], [sg * 0.1, 0.27, 0], 0.05, 0.062, S, 7, false);
    ball(M, [sg * 0.1, 0.285, 0.015], 0.068, S2, { n: 6, k: 3 });
    limb(M, [sg * 0.1, 0.29, 0], [sg * 0.105, 0.48, 0], 0.068, 0.088, S, 7, false);
    ball(M, [sg * 0.235, 0.795, 0], 0.09, S2, { n: 8, k: 4, sy: 0.8 });
  }
  lathe(M, [[0.165, 0.43], [0.195, 0.47], [0.16, 0.55]], S2, { n: 10, sz: 0.74 });
  lathe(M, [[0.15, 0.52], [0.14, 0.57], [0.17, 0.63], [0.205, 0.71], [0.2, 0.77], [0.12, 0.825], [0.07, 0.84]], S, { n: 10, sz: 0.74 });
  lathe(M, [[0.142, 0.565], [0.15, 0.575], [0.142, 0.585]], GOLD, { n: 10, sz: 0.76 });
  lathe(M, [[0.07, 0.83], [0.08, 0.875]], S2, { n: 8 });
  lathe(M, [[0.074, 0.865], [0.096, 0.89], [0.102, 0.94], [0.092, 0.985], [0.05, 1.015], [0, 1.022]], S, { n: 10 });
  ball(M, [0, 0.915, 0.075], 0.05, S2, { n: 6, k: 3, sz: 1.35, sy: 0.8 });
  pbox(P, -0.062, 0.062, 0.94, 0.952, 0.085, 0.108, BLACK);
  limb(P, [0, 1.012, -0.01], [0, 1.09, -0.09], 0.022, 0.045, RED, 6); ball(P, [0, 1.1, -0.12], 0.05, RED, { n: 6, k: 3, sy: 1.3 });
  if (pose === 0) {
    for (const sg of [-1, 1]) arm(M, [sg * 0.245, 0.78, 0], [sg * 0.27, 0.63, 0.06], [sg * 0.045, 0.6, 0.2], S, 0.048);
    both([M], [0, 0.56, 0.2], 0, 1, Math.PI, () => sword(M, 0.56));
  } else {
    arm(M, [0.245, 0.78, 0], [0.3, 0.64, 0.04], [0.31, 0.62, 0.17], S, 0.048);
    arm(M, [-0.245, 0.78, 0], [-0.275, 0.62, 0.02], [-0.265, 0.46, 0.06], S, 0.048);
    both([M, P], [0.31, 0, 0.17], 0, 1, 0, () => halberd(M, P, 1.3));
    slab(M, [[-0.36, 0.72], [-0.16, 0.72], [-0.16, 0.56], [-0.26, 0.42], [-0.36, 0.56]], 0.1, 0.12, RED.map((q) => q * 1.2), IRON);
  }
}

// a carved figure, 1 unit tall
function statue(K, kind) {
  const P = K.plain, C = STONE, C2 = STONE2;
  const robe = () => { lathe(P, [[0.21, 0], [0.2, 0.03], [0.155, 0.3], [0.135, 0.46], [0.165, 0.53]], C, { n: 10, sz: 0.8 }); for (let i = 0; i < 5; i++) { const a = (i - 2) * 0.5; limb(P, [Math.sin(a) * 0.19, 0.02, Math.cos(a) * 0.15], [Math.sin(a) * 0.13, 0.5, Math.cos(a) * 0.1], 0.02, 0.012, C2, 4, false); } };
  const torso = () => { lathe(P, [[0.155, 0.5], [0.145, 0.56], [0.17, 0.66], [0.19, 0.74], [0.11, 0.8], [0.062, 0.82], [0.058, 0.86]], C, { n: 10, sz: 0.7 }); for (const sg of [-1, 1]) ball(P, [sg * 0.2, 0.765, 0], 0.07, C, { n: 6, k: 3 }); };
  const head = () => { ball(P, [0, 0.9, 0], 0.07, C, { n: 8, k: 4, sy: 1.12 }); pbox(P, -0.012, 0.012, 0.88, 0.915, 0.062, 0.082, C2); };
  if (kind === 'king') {
    robe(); torso(); head();
    ball(P, [0, 0.855, 0.045], 0.05, C2, { n: 6, k: 3, sy: 1.3 });                                  // beard
    lathe(P, [[0.07, 0.94], [0.076, 0.945], [0.09, 1.0]], C, { n: 8 });
    for (let i = 0; i < 8; i++) { const a = i * 0.785; limb(P, [Math.cos(a) * 0.088, 0.995, Math.sin(a) * 0.088], [Math.cos(a) * 0.096, 1.035, Math.sin(a) * 0.096], 0.014, 0, C, 4, false); }
    lathe(P, [[0.21, 0.78], [0.23, 0.6], [0.26, 0.05]], C2, { n: 10, sz: 0.45, cz: -0.07 });       // the cloak behind
    for (const sg of [-1, 1]) arm(P, [sg * 0.215, 0.76, 0], [sg * 0.25, 0.62, 0.07], [sg * 0.04, 0.6, 0.2], C, 0.042);
    both([P], [0, 0.56, 0.2], 0, 1, Math.PI, () => sword(P, 0.56, C2, C2));
  } else if (kind === 'knight') {
    for (const sg of [-1, 1]) { limb(P, [sg * 0.09, 0, 0], [sg * 0.095, 0.5, 0], 0.062, 0.085, C, 7); ball(P, [sg * 0.09, 0.03, 0.045], 0.06, C, { n: 6, k: 3, sy: 0.6, sz: 1.6 }); }
    lathe(P, [[0.185, 0.3], [0.165, 0.52]], C2, { n: 10, sz: 0.72 });
    torso();
    lathe(P, [[0.072, 0.83], [0.084, 0.86], [0.086, 0.975], [0.06, 0.992], [0, 0.995]], C, { n: 8 });
    pbox(P, -0.06, 0.06, 0.925, 0.937, 0.07, 0.092, BLACK); pbox(P, -0.008, 0.008, 0.86, 0.975, 0.078, 0.092, C2);
    arm(P, [0.215, 0.76, 0], [0.26, 0.62, 0.04], [0.25, 0.55, 0.17], C, 0.042);
    both([P], [0.25, 0.55, 0.19], 0, 1, Math.PI, () => sword(P, 0.55, C2, C2));
    arm(P, [-0.215, 0.76, 0], [-0.27, 0.62, 0.06], [-0.2, 0.56, 0.15], C, 0.042);
    slab(P, [[-0.34, 0.74], [-0.08, 0.74], [-0.08, 0.52], [-0.21, 0.34], [-0.34, 0.52]], 0.16, 0.19, C, C2);
    pbox(P, -0.225, -0.195, 0.4, 0.72, 0.19, 0.2, C2); pbox(P, -0.31, -0.11, 0.6, 0.63, 0.19, 0.2, C2);
  } else {                                                                                         // a bishop
    robe(); torso(); head();
    lathe(P, [[0.072, 0.945], [0.088, 1.0], [0.0, 1.12]], C, { n: 8, sz: 0.55 });
    lathe(P, [[0.2, 0.78], [0.21, 0.62], [0.2, 0.42]], C2, { n: 10, sz: 0.5, cz: -0.05 });
    arm(P, [0.215, 0.76, 0], [0.27, 0.64, 0.05], [0.27, 0.66, 0.16], C, 0.042);
    limb(P, [0.27, 0, 0.17], [0.27, 1.1, 0.17], 0.016, 0.014, C2, 5);
    for (let i = 0; i < 6; i++) { const a0 = i * 0.8 - 0.4, a1 = a0 + 0.8; limb(P, [0.27 - 0.06 + Math.cos(a0) * 0.06, 1.1 + Math.sin(a0) * 0.06, 0.17], [0.27 - 0.06 + Math.cos(a1) * 0.06, 1.1 + Math.sin(a1) * 0.06, 0.17], 0.013, 0.011, C2, 4); }
    arm(P, [-0.215, 0.76, 0], [-0.26, 0.64, 0.07], [-0.2, 0.78, 0.15], C, 0.042);
  }
}

const HEATER = [[-0.5, 0.6], [0.5, 0.6], [0.5, 0.12], [0.42, -0.2], [0.25, -0.45], [0, -0.6], [-0.25, -0.45], [-0.42, -0.2], [-0.5, 0.12]];
function shieldFace(K, k) {
  const r = CELL.shield(k), uv = (p) => [r[0] + (p[0] + 0.5) * (r[2] - r[0]), r[1] + (p[1] + 0.6) / 1.2 * (r[3] - r[1])];
  slab(K.metal, HEATER.map((p) => [p[0] * 1.07, p[1] * 1.06]), 0.07, 0.125, IRON);
  for (let i = 0; i < HEATER.length; i++) { const a = HEATER[i], b = HEATER[(i + 1) % HEATER.length]; K.cloth.tri([0, 0, 0.14], [a[0], a[1], 0.13], [b[0], b[1], 0.13], [1, 1, 1], [0, 0, 1], null, [uv([0, 0]), uv(a), uv(b)]); }
}

export function buildProp(d, K) {
  const M = K.metal, P = K.plain, T = K.cloth;
  switch (d.t) {
    case 'armor': both([M, P], [d.x, d.y + 0.22 * d.s, d.z], d.yaw, d.s, 0, () => armor(K, d.pose)); break;
    case 'statue': {
      const w = d.s * 0.21, p = d.ped;
      both([P], [d.x, d.y, d.z], d.yaw, 1, 0, () => { pbox(P, -w * 1.25, w * 1.25, 0, p * 0.14, -w * 1.25, w * 1.25, STONE2); pbox(P, -w, w, p * 0.14, p * 0.86, -w, w, STONE); pbox(P, -w * 1.2, w * 1.2, p * 0.86, p, -w * 1.2, w * 1.2, STONE2); });
      both([P], [d.x, d.y + p, d.z], d.yaw, d.s, 0, () => statue(K, d.kind));
      break;
    }
    case 'shield':
      both([M, P, T], [d.x, d.y, d.z], yawOf(d.nx, d.nz), d.s, 0, () => {
        if (d.arms === 'swords') for (const sg of [-1, 1]) both([M], [sg * 0.45, -0.55, 0.045 + sg * 0.012], 0, 1, sg * 0.69, () => sword(M, 1.3));
        if (d.arms === 'halberds') for (const sg of [-1, 1]) both([M, P], [sg * 0.55, -1.0, 0.045 + sg * 0.012], 0, 1, sg * 0.5, () => halberd(M, P, 1.95));
        shieldFace(K, d.k);
      });
      break;
    case 'banner': {
      const r = CELL.banner(d.k), w = d.w, h = d.h, n = 5, off = d.free ? 0 : 0.12, tail = 0.4 * w;
      both([M, T], [d.x, d.y, d.z], yawOf(d.nx, d.nz), 1, 0, () => {
        const uv = (x, y) => [r[0] + (x / w + 0.5) * (r[2] - r[0]), r[3] + y / h * (r[3] - r[1])], zz = (y) => off + 0.07 * Math.sin(y * 1.9 + d.k);
        for (let i = 0; i < n; i++) {
          const y0 = -h * i / n, y1 = -h * (i + 1) / n, last = i === n - 1, ym = last ? y1 + tail : y1;
          for (const [xa, xb, ya, yb] of [[-w / 2, 0, y1, ym], [0, w / 2, ym, y1]]) T.quad([xa, ya, zz(y1)], [xb, yb, zz(y1)], [xb, y0, zz(y0)], [xa, y0, zz(y0)], [1, 1, 1], [0, 0, 1], null, [uv(xa, ya), uv(xb, yb), uv(xb, y0), uv(xa, y0)]);
        }
        if (!d.free) { limb(M, [-w / 2 - 0.2, 0.05, off], [w / 2 + 0.2, 0.05, off], 0.05, 0.05, IRON, 5); for (const sg of [-1, 1]) { limb(M, [sg * (w / 2 + 0.1), 0.05, 0], [sg * (w / 2 + 0.1), 0.05, off], 0.035, 0.035, IRON, 4); ball(M, [sg * (w / 2 + 0.25), 0.05, off], 0.09, GOLD, { n: 6, k: 3 }); } }
      });
      break;
    }
    case 'tapestry': {
      const r = CELL.tapestry(d.k), w = d.w, h = d.h;
      both([P, T, M], [d.x, d.y, d.z], yawOf(d.nx, d.nz), 1, 0, () => {
        T.quadUV([-w / 2, -h, 0.1], [w / 2, -h, 0.1], [w / 2, 0, 0.08], [-w / 2, 0, 0.08], r, [0, 0, 1]);
        limb(P, [-w / 2 - 0.25, 0.06, 0.1], [w / 2 + 0.25, 0.06, 0.1], 0.07, 0.07, DARK, 6);
        for (const sg of [-1, 1]) ball(M, [sg * (w / 2 + 0.3), 0.06, 0.1], 0.11, GOLD, { n: 6, k: 3 });
      });
      break;
    }
    case 'carpet': {
      const r = CELL.carpet(d.k), y = 0.11, ax = d.ax === 'x', a0 = ax ? d.x0 : d.z0, a1 = ax ? d.x1 : d.z1, b0 = ax ? d.z0 : d.x0, b1 = ax ? d.z1 : d.x1, wd = b1 - b0;
      for (let a = a0; a < a1 - 0.01; a += wd) {
        const e = Math.min(a + wd, a1), f = (e - a) / wd, rr = [r[0], r[1], r[2], r[1] + (r[3] - r[1]) * f];
        if (ax) T.quadUV([a, y, b1], [a, y, b0], [e, y, b0], [e, y, b1], rr, [0, 1, 0]); else T.quadUV([b0, y, a], [b1, y, a], [b1, y, e], [b0, y, e], rr, [0, 1, 0]);
      }
      break;
    }
    case 'glass': {
      const r = CELL.glass(d.k), w = d.w, h = d.h;
      both([K.glass], [d.x, d.y, d.z], yawOf(d.nx, d.nz), 1, 0, () => K.glass.quadUV([-w / 2, -h / 2, 0.05], [w / 2, -h / 2, 0.05], [w / 2, h / 2, 0.05], [-w / 2, h / 2, 0.05], r, [0, 0, 1]));
      break;
    }
    case 'rib': {
      const B = K.stone, col = [1.12, 1.12, 1.1], N = 14, ax = d.axis === 'x', ro = d.hw + 0.06, ri = d.hw - 0.42, w0 = d.w - 0.45, w1 = d.w + 0.45, wv = B.wv;
      const pt = (rad, a, w) => (ax ? [w, d.ys + rad * Math.sin(a), d.c + rad * Math.cos(a)] : [d.c + rad * Math.cos(a), d.ys + rad * Math.sin(a), w]);
      B.wv = 0;
      for (let k = 0; k < N; k++) {
        const a = Math.PI * k / N, b = Math.PI * (k + 1) / N, m = (a + b) / 2, inw = ax ? [0, -Math.sin(m), -Math.cos(m)] : [-Math.cos(m), -Math.sin(m), 0];
        B.quad(pt(ri, a, w0), pt(ri, a, w1), pt(ri, b, w1), pt(ri, b, w0), col, inw);
        B.quad(pt(ri, a, w0), pt(ro, a, w0), pt(ro, b, w0), pt(ri, b, w0), col, ax ? [-1, 0, 0] : [0, 0, -1]);
        B.quad(pt(ri, a, w1), pt(ro, a, w1), pt(ro, b, w1), pt(ri, b, w1), col, ax ? [1, 0, 0] : [0, 0, 1]);
      }
      B.wv = wv;
      break;
    }
    case 'chandelier': {
      const n = 10, r = d.r;
      both([M, P, K.fire], [d.x, d.y, d.z], 0, 1, 0, () => {
        lathe(M, [[r - 0.1, -0.13], [r + 0.1, -0.13], [r + 0.1, 0.13], [r - 0.1, 0.13], [r - 0.1, -0.13]], IRON, { n: 20 });
        lathe(M, [[r + 0.1, -0.04], [r + 0.13, 0], [r + 0.1, 0.04]], GOLD, { n: 20 });
        for (let i = 0; i < 4; i++) { const a = i * 1.5708 + 0.4, c = Math.cos(a), s = Math.sin(a); limb(M, [0, 0, 0], [c * r, 0, s * r], 0.05, 0.05, IRON, 4, false); limb(M, [c * r, 0.1, s * r], [0, r * 0.95, 0], 0.03, 0.03, IRON, 4, false); }
        ball(M, [0, 0, 0], 0.2, GOLD, { n: 8, k: 4 }); limb(M, [0, r * 0.95, 0], [0, d.top - d.y, 0], 0.035, 0.035, IRON, 4, false);
        for (let i = 0; i < n; i++) {
          const a = i / n * 6.2832, x = Math.cos(a) * r, z = Math.sin(a) * r;
          lathe(M, [[0.05, 0.13], [0.13, 0.2], [0.11, 0.22]], GOLD, { n: 6, cx: x, cz: z });
          limb(P, [x, 0.2, z], [x, 0.62, z], 0.05, 0.05, WAX, 5);
          addCyl(K.fire, { x, z, r: 0.06, r1: 0, y0: 0.62, y1: 0.95, n: 5 }, FL1);
        }
      });
      K.lights.push({ x: d.x, y: d.y + 0.7, z: d.z, I: 1.7, halo: 3.2 });
      break;
    }
    case 'torch': {
      const { x, y, z, nx = 0, nz = 0 } = d;
      limb(M, [x - nx * 0.42, y - 1.15, z - nz * 0.42], [x, y - 0.62, z], 0.05, 0.045, IRON, 5);
      lathe(M, [[0.05, y - 0.66], [0.19, y - 0.52], [0.24, y - 0.2], [0.2, y - 0.2]], IRON, { n: 8, cx: x, cz: z });
      if (nx || nz) pbox(M, x - nx * 0.42 - 0.12 - Math.abs(nz) * 0.1, x - nx * 0.42 + 0.12 + Math.abs(nz) * 0.1, y - 1.45, y - 0.95, z - nz * 0.42 - 0.12 - Math.abs(nx) * 0.1, z - nz * 0.42 + 0.12 + Math.abs(nx) * 0.1, IRON);
      addCyl(K.fire, { x, z, r: 0.23, r1: 0, y0: y - 0.25, y1: y + 0.62, n: 7 }, FL0);
      addCyl(K.fire, { x, z, r: 0.12, r1: 0, y0: y - 0.22, y1: y + 0.3, n: 5 }, FL1);
      K.lights.push({ x: x + nx * 0.25, y: y + 0.25, z: z + nz * 0.25, I: 1.0, halo: 2.6 });
      break;
    }
    case 'brazier':
      both([M, K.fire], [d.x, d.y, d.z], 0, 1.25, 0, () => {
        for (let i = 0; i < 3; i++) { const a = i * 2.094 + 0.5; limb(M, [Math.cos(a) * 0.55, 0, Math.sin(a) * 0.55], [Math.cos(a) * 0.3, 1.12, Math.sin(a) * 0.3], 0.04, 0.035, IRON, 5); }
        lathe(M, [[0.1, 1.0], [0.4, 1.1], [0.55, 1.33], [0.58, 1.55], [0.5, 1.55], [0.46, 1.36], [0, 1.3]], IRON, { n: 10 });
        lathe(M, [[0.58, 1.5], [0.61, 1.53], [0.58, 1.56]], GOLD, { n: 10 });
        ball(K.fire, [0, 1.5, 0], 0.46, [0.5, 0.12, 0.03], { n: 8, k: 3, sy: 0.4 });
        addCyl(K.fire, { x: 0, z: 0, r: 0.34, r1: 0, y0: 1.5, y1: 2.6, n: 7 }, FL0); addCyl(K.fire, { x: 0, z: 0, r: 0.18, r1: 0, y0: 1.5, y1: 2.1, n: 5 }, FL1);
        addCyl(K.fire, { x: 0.22, z: 0.1, r: 0.18, r1: 0, y0: 1.5, y1: 2.1, n: 5 }, FL0); addCyl(K.fire, { x: -0.19, z: -0.14, r: 0.18, r1: 0, y0: 1.5, y1: 2.25, n: 5 }, FL0);
      });
      K.lights.push({ x: d.x, y: d.y + 2.7, z: d.z, I: 1.6, halo: 4 });
      break;
    case 'candle':
      lathe(M, [[0.16, d.y], [0.05, d.y + 0.06], [0.04, d.y + 0.5], [0.12, d.y + 0.56]], GOLD, { n: 6, cx: d.x, cz: d.z });
      limb(P, [d.x, d.y + 0.56, d.z], [d.x, d.y + 1.2, d.z], 0.055, 0.055, WAX, 5);
      addCyl(K.fire, { x: d.x, z: d.z, r: 0.07, r1: 0, y0: d.y + 1.2, y1: d.y + 1.55, n: 5 }, FL1);
      K.lights.push({ x: d.x, y: d.y + 1.5, z: d.z, I: 0.6, halo: 1.6 });
      break;
    case 'barrel':
      both([P, M], [d.x, d.y, d.z], d.x * 3.1, d.s, 0, () => {
        lathe(P, [[0, 0], [0.38, 0], [0.45, 0.3], [0.48, 0.6], [0.45, 0.9], [0.38, 1.2], [0, 1.2]], OAK, { n: 10 });
        for (const y of [0.14, 0.42, 0.78, 1.06]) { const rr = 0.49 - Math.abs(y - 0.6) * 0.19; lathe(M, [[rr, y - 0.03], [rr + 0.012, y], [rr, y + 0.03]], IRON, { n: 10 }); }
      });
      break;
    case 'crate': {
      const h = d.s / 2;
      both([P], [d.x, d.y, d.z], d.yaw || d.x * 1.7, 1, 0, () => {
        pbox(P, -h, h, 0, d.s, -h, h, OAK);
        for (const sg of [-1, 1]) { pbox(P, -h - 0.02, h + 0.02, 0, d.s + 0.02, sg * h - 0.06, sg * h + 0.06, DARK); pbox(P, sg * h - 0.06, sg * h + 0.06, 0, d.s + 0.02, -h - 0.02, h + 0.02, DARK); }
        pbox(P, -h - 0.02, h + 0.02, d.s - 0.1, d.s + 0.02, -h - 0.02, h + 0.02, DARK);
      });
      break;
    }
    case 'chest':
      both([P, M], [d.x, d.y, d.z], d.yaw, 1, 0, () => {
        pbox(P, -0.95, 0.95, 0, 0.7, -0.5, 0.5, OAK);
        for (let i = 0; i < 6; i++) { const a0 = i / 6 * Math.PI, a1 = (i + 1) / 6 * Math.PI; P.quad([-0.95, 0.7 + Math.sin(a0) * 0.32, Math.cos(a0) * 0.5], [0.95, 0.7 + Math.sin(a0) * 0.32, Math.cos(a0) * 0.5], [0.95, 0.7 + Math.sin(a1) * 0.32, Math.cos(a1) * 0.5], [-0.95, 0.7 + Math.sin(a1) * 0.32, Math.cos(a1) * 0.5], DARK, [0, Math.sin((a0 + a1) / 2), Math.cos((a0 + a1) / 2)]); }
        for (const sg of [-1, 1]) for (let i = 0; i < 6; i++) { const a0 = i / 6 * Math.PI, a1 = (i + 1) / 6 * Math.PI; P.tri([sg * 0.95, 0.7, 0], [sg * 0.95, 0.7 + Math.sin(a0) * 0.32, Math.cos(a0) * 0.5], [sg * 0.95, 0.7 + Math.sin(a1) * 0.32, Math.cos(a1) * 0.5], DARK, [sg, 0, 0]); }
        for (const x of [-0.7, 0, 0.7]) { pbox(M, x - 0.06, x + 0.06, 0, 0.72, -0.52, 0.52, IRON); }
        pbox(M, -0.1, 0.1, 0.48, 0.74, 0.5, 0.56, GOLD);
      });
      break;
    case 'rack': {
      const w = d.w, n = Math.round(w / 0.75);
      both([P, M], [d.x, d.y, d.z], yawOf(d.nx, d.nz), 1, 0, () => {
        pbox(P, -w / 2, w / 2, 0, 0.22, 0.1, 0.75, DARK);
        for (const y of [1.2, 2.5]) pbox(P, -w / 2, w / 2, y, y + 0.16, 0.12, 0.3, OAK);
        for (const sg of [-1, 1]) pbox(P, sg * w / 2 - 0.1, sg * w / 2 + 0.1, 0, 2.9, 0.08, 0.32, DARK);
        for (let i = 0; i < n; i++) {
          const x = -w / 2 + (i + 0.5) * w / n;
          if (i % 3 === 2) both([M], [x, 2.2, 0.36], 0, 2.0, Math.PI, () => sword(M, 0.95));
          else both([M, P], [x, 0.22, 0.42], (i % 2 ? 0.3 : -0.2), 2.5, 0, () => halberd(M, P, 1.3 + (i % 2) * 0.1));
        }
      });
      break;
    }
    case 'grate': {
      const ax = d.axis === 'x', pt = (u, y, e = 0) => (ax ? [d.w + e, y, d.c + u] : [d.c + u, y, d.w + e]);
      for (let u = -d.hw + 0.5; u < d.hw; u += 1) {
        const top = d.ys + Math.sqrt(Math.max(0, d.hw * d.hw - u * u)) + 0.05; if (top < d.yb + 0.2) continue;
        limb(M, pt(u, d.yb), pt(u, top), 0.07, 0.07, IRON, 4, false); limb(M, pt(u, d.yb), pt(u, d.yb - 0.5), 0.09, 0, IRON, 4, false);
      }
      const hy = d.yb + 0.7, hu = Math.sqrt(Math.max(0, d.hw * d.hw - (hy - d.ys) * (hy - d.ys)));
      limb(M, pt(-hu, hy), pt(hu, hy), 0.07, 0.07, IRON, 4, false);
      break;
    }
    case 'tree': {
      const h = d.h, G = [[0.035, 0.07, 0.018], [0.05, 0.09, 0.022], [0.028, 0.055, 0.016], [0.06, 0.1, 0.03]], BARK = [0.085, 0.068, 0.05];
      both([P], [d.x, d.y, d.z], 0.6, 1, 0, () => {
        limb(P, [0, 0, 0], [0.2, h * 0.5, 0.1], 0.85, 0.5, BARK, 8, false);
        lathe(P, [[1.5, 0], [0.9, 0.5], [0.85, 1.2]], BARK, { n: 8 });
        for (const [a, l] of [[0.3, 4.5], [2.2, 4], [4.2, 4.8]]) limb(P, [0.1, h * 0.4, 0.05], [Math.cos(a) * l, h * 0.62, Math.sin(a) * l], 0.3, 0.14, BARK, 5, false);
        [[0, 0.72, 0, 5.2], [3.4, 0.64, 1, 3.8], [-2.8, 0.66, 2.4, 3.9], [-1.2, 0.62, -3.4, 3.7], [2.6, 0.7, -2.6, 3.4], [0.4, 0.88, 0.6, 3.6], [-3.2, 0.75, -0.6, 3.0], [4.6, 0.56, -0.8, 2.2], [-4.4, 0.57, 1.6, 2.4], [1.0, 0.56, 4.4, 2.3], [-0.6, 0.55, -4.6, 2.2], [2.2, 0.93, -0.8, 2.4], [-1.6, 0.9, 1.2, 2.5]].forEach(([x, f, z, r], i) => ball(P, [x, h * f, z], r, G[i % 4], { n: 9, k: 5, sy: 0.8 }));
      });
      break;
    }
    default: return false;
  }
  return true;
}

// painted shutters either side of a window (white and red flames, as on many a German castle)
export function shutters(d, K) {
  const r = CELL.shutter(), w = d.w * 0.55, h = d.h * 0.86;
  both([K.cloth], [d.x, d.y - d.h * 0.07, d.z], yawOf(d.nx, d.nz), 1, 0, () => {
    for (const sg of [-1, 1]) { const x0 = sg * (d.w / 2 + 0.08), x1 = x0 + sg * w; K.cloth.quadUV([Math.min(x0, x1), -h / 2, 0.06], [Math.max(x0, x1), -h / 2, 0.06], [Math.max(x0, x1), h / 2, 0.06], [Math.min(x0, x1), h / 2, 0.06], r, [0, 0, 1]); }
  });
}
