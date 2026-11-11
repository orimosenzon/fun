// Blade-element aerodynamics for a man with wings.
// Each wing is cut into spanwise strips. Every strip sees its own air: the body's velocity,
// the body's rotation, the wing's own flapping motion, the local wind and the downwash the
// wings themselves leave behind (momentum theory). From that local flow each strip gets an
// angle of attack, and from a full-range lift/drag curve (attached flow, stall, flat plate) a force.
// Nothing else is scripted: gliding, stalling, thrust from the downstroke, induced drag,
// ground effect and roll into a thermal all fall out of the same sum.
// Body axes: x forward (head), y up (back), z right. Units SI.
export const RHO = 1.225, G = 9.81, MASS = 86;
export const SHOULDER = [0.42, 0.08, 0.2];
export const S0 = [0.45, 1.15, 1.85, 2.55, 3.25, 3.95], STRIP_W = 0.7, CHORD = [1.45, 1.5, 1.45, 1.3, 1.05, 0.7];
export const SEMI = 4.3;
export const WING_AREA = 2 * STRIP_W * CHORD.reduce((a, b) => a + b, 0);
const CAMBER = 0.075, INCIDENCE = 0.035;
export const spanFactor = (fold) => 0.26 + 0.74 * fold;
export const halfSpan = (fold) => SHOULDER[2] + SEMI * spanFactor(fold);

export function newAcc() { return { fx: 0, fy: 0, fz: 0, tx: 0, ty: 0, tz: 0, power: 0, fn: 0, lx: 0, ly: 0, lz: 0, stall: 0 }; }
export function clearAcc(a) { a.fx = a.fy = a.fz = a.tx = a.ty = a.tz = a.power = a.fn = a.stall = 0; }

// lift and drag coefficients over the full circle. a is measured from the zero-lift line.
export function liftDrag(a, ks = 1) {
  const sa = Math.sin(a), ca = Math.cos(a);
  const k = 1 / (1 + Math.exp((Math.abs(a) - 0.32) * 40)); // 1 attached .. 0 separated
  return [k * 5.7 * ks * a + (1 - k) * 1.9 * sa * ca, k * (0.012 + 0.25 * a * a) + (1 - k) * (0.02 + 1.9 * sa * sa), k];
}

// one flat strip at r with chord direction c and normal n, moving at v through still air
function strip(acc, rx, ry, rz, cx, cy, cz, nx, ny, nz, vx, vy, vz, area, camber, ks) {
  const vc = vx * cx + vy * cy + vz * cz, vn = vx * nx + vy * ny + vz * nz;
  const V2 = vc * vc + vn * vn;
  if (V2 < 1e-6) { acc.lx = acc.ly = acc.lz = 0; return; }
  const V = Math.sqrt(V2), a = Math.atan2(-vn, vc) + camber;
  const sa = Math.sin(a), ca = Math.cos(a);
  const k = 1 / (1 + Math.exp((Math.abs(a) - 0.32) * 40));
  const cl = k * 5.7 * ks * a + (1 - k) * 1.9 * sa * ca;
  const cd = k * (0.012 + 0.25 * a * a) + (1 - k) * (0.02 + 1.9 * sa * sa);
  const q = 0.5 * RHO * V * area;
  const Fc = -q * (cl * vn + cd * vc), Fn = q * (cl * vc - cd * vn);
  const fx = Fc * cx + Fn * nx, fy = Fc * cy + Fn * ny, fz = Fc * cz + Fn * nz;
  acc.fx += fx; acc.fy += fy; acc.fz += fz;
  acc.tx += ry * fz - rz * fy; acc.ty += rz * fx - rx * fz; acc.tz += rx * fy - ry * fx;
  acc.lx = fx; acc.ly = fy; acc.lz = fz;
  acc.stall += (1 - k) * area;
}

// One wing. side = +1 right, -1 left.
// w: { phi (flap angle, up positive), phiDot, fold (0 tucked .. 1 spread), sweep (rad, back positive), twist (rad), eta (how far the feathers twist into the stroke's own wind), unload (extra nose-down twist of the outer wing on the upstroke) }
// vb: body velocity through the air mass, om: body rates, dw: extra wind at this wing (body frame), vi: downwash speed
export function wingAero(acc, side, w, vb, om, dw, vi) {
  const sf = spanFactor(w.fold), cs = 0.72 + 0.28 * w.fold;
  const lam = w.sweep + (1 - w.fold) * 0.8;
  const cl = Math.cos(lam), sl = Math.sin(lam), cp = Math.cos(w.phi), sp = Math.sin(w.phi);
  const d0 = -sl * cp, d1 = sp, d2 = side * cl * cp; // span direction
  const e0 = sl * sp, e1 = cp, e2 = -side * cl * sp; // d(span)/d(phi)
  let c0x = 1 - d0 * d0, c0y = -d1 * d0, c0z = -d2 * d0;
  const cn = 1 / Math.hypot(c0x, c0y, c0z); c0x *= cn; c0y *= cn; c0z *= cn;
  const n0x = side * (d1 * c0z - d2 * c0y), n0y = side * (d2 * c0x - d0 * c0z), n0z = side * (d0 * c0y - d1 * c0x);
  const Vf = Math.max(vb[0], 6);
  const sx = SHOULDER[0], sy = SHOULDER[1], sz = side * SHOULDER[2];
  for (let i = 0; i < 6; i++) {
    const s = S0[i] * sf, u = S0[i] / SEMI, ch = CHORD[i] * cs;
    const back = 0.25 * ch;
    const rx = sx + d0 * s - c0x * back, ry = sy + d1 * s - c0y * back, rz = sz + d2 * s - c0z * back;
    const fv = s * w.phiDot; // flapping speed of this strip
    const th = INCIDENCE - 0.05 * u * u + w.twist * (0.3 + 0.7 * u) + w.eta * Math.atan2(fv, Vf) - w.unload * u;
    const ct = Math.cos(th), st = Math.sin(th);
    const cx = c0x * ct + n0x * st, cy = c0y * ct + n0y * st, cz = c0z * ct + n0z * st;
    const nx = n0x * ct - c0x * st, ny = n0y * ct - c0y * st, nz = n0z * ct - c0z * st;
    const vx = vb[0] + om[1] * rz - om[2] * ry + e0 * fv - dw[0] + n0x * vi;
    const vy = vb[1] + om[2] * rx - om[0] * rz + e1 * fv - dw[1] + n0y * vi;
    const vz = vb[2] + om[0] * ry - om[1] * rx + e2 * fv - dw[2] + n0z * vi;
    strip(acc, rx, ry, rz, cx, cy, cz, nx, ny, nz, vx, vy, vz, STRIP_W * sf * ch, CAMBER, 1);
    acc.power -= (acc.lx * e0 + acc.ly * e1 + acc.lz * e2) * fv;
    acc.fn += acc.lx * n0x + acc.ly * n0y + acc.lz * n0z;
  }
}

// tail fan between the legs (pitch), the fin strapped to his calves (yaw), and the body itself
export function bodyAero(acc, tail, vb, om, vi, legs) {
  let rx = -1.15, ci = Math.cos(tail.inc), si = Math.sin(tail.inc);
  let vx = vb[0], vy = vb[1] + om[2] * rx + vi, vz = vb[2] - om[1] * rx;
  const st0 = acc.stall;
  strip(acc, rx, 0, 0, ci, si, 0, -si, ci, 0, vx, vy, vz, 0.95, 0, 0.5);
  rx = -1.3; ci = Math.cos(tail.rud); si = Math.sin(tail.rud);
  vy = vb[1] + om[2] * rx; vz = vb[2] - om[1] * rx;
  strip(acc, rx, 0, 0, ci, 0, -si, si, 0, ci, vx, vy, vz, 1.0, 0, 0.7);
  acc.stall = st0;
  // bluff body: prone and head-first it is slippery, belly-first it is a parachute
  rx = -0.12;
  vy = vb[1] + om[2] * rx; vz = vb[2] - om[1] * rx;
  const V = Math.hypot(vx, vy, vz), k = -0.5 * RHO * V;
  const fx = k * vx * (0.15 + 0.3 * legs), fy = k * vy * 0.75, fz = k * vz * 0.42;
  acc.fx += fx; acc.fy += fy; acc.fz += fz;
  acc.ty += -rx * fz; acc.tz += rx * fy;
  // rotational damping of the body and limbs
  acc.tx -= 2.0 * om[0]; acc.ty -= 4.0 * om[1]; acc.tz -= 4.0 * om[2];
}
