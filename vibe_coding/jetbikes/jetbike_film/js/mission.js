// Flight plan -> smooth reference trajectory (position, velocity, acceleration, heading).
// The plan is a 3D spline through waypoints; a speed profile is planned along it with
// limits on longitudinal acceleration, braking and lateral (turn) acceleration, then the
// time series is smoothed so the reference jerk stays finite. The flight computer tracks
// this reference; the vehicle's real motion comes out of the physics.

import { heightAt, surfaceAt, valleyZ, LAKE, HILL, PAD, LAND } from './terrain.js';
import { clamp, lerp } from './vmath.js';

export const TL = {
  liftStart: [1.2, 2.0, 2.8, 3.6],  // lift turbojets light one after another
  mainStart: 4.6,
  spoolUp: 8.2,                     // collective ramp on the ground
  liftoff: 10.6,                    // flight mode engaged
};

const HOVER_AGL = 4.5;
const CRUISE = 44;
const A_ACC = 2.8, A_DEC = 2.7, A_LAT = 5.2;
const DT = 0.01;

function catmull(P, n) {
  // centripetal Catmull-Rom through P (with duplicated ends)
  const pts = [P[0], ...P, P[P.length - 1]];
  const out = [];
  for (let i = 1; i < pts.length - 2; i++) {
    const [p0, p1, p2, p3] = [pts[i - 1], pts[i], pts[i + 1], pts[i + 2]];
    const d = (a, b) => Math.pow(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) + 1e-6, 0.5);
    const t0 = 0, t1 = t0 + d(p0, p1), t2 = t1 + d(p1, p2), t3 = t2 + d(p2, p3);
    for (let s = 0; s < n; s++) {
      const t = t1 + (t2 - t1) * s / n;
      const L = (a, b, ta, tb) => a.map((_, k) => ((tb - t) * a[k] + (t - ta) * b[k]) / (tb - ta));
      const A1 = L(p0, p1, t0, t1), A2 = L(p1, p2, t1, t2), A3 = L(p2, p3, t2, t3);
      const B1 = L(A1, A2, t0, t2), B2 = L(A2, A3, t1, t3);
      out.push(L(B1, B2, t1, t2));
    }
  }
  out.push(P[P.length - 1]);
  return out;
}

function gaussSmooth(arr, sigmaSamples) {
  const r = Math.ceil(sigmaSamples * 3);
  const w = [];
  for (let k = -r; k <= r; k++) w.push(Math.exp(-0.5 * (k / sigmaSamples) ** 2));
  const n = arr.length;
  return arr.map((_, i) => {
    let s = 0, ws = 0;
    for (let k = -r; k <= r; k++) {
      const j = clamp(i + k, 0, n - 1);
      s += arr[j] * w[k + r]; ws += w[k + r];
    }
    return s / ws;
  });
}

export function buildMission(cgHeight) {
  const lakeZ = valleyZ(LAKE.x) + LAKE.dz;
  const hz = valleyZ(HILL.x) + HILL.dz;
  const R = 190;
  const g = cgHeight;
  // [x, z, AGL, speed cap]
  const wp = [
    [0, 0, HOVER_AGL, 3],
    [45, 1, 6, 16],
    [170, valleyZ(170) - 8, 17, CRUISE],
    [330, valleyZ(330), 26, CRUISE],
    [470, lakeZ + 12, 11, 40],
    [560, lakeZ + 8, 4.5, 38, 1],
    [650, lakeZ + 4, 4.0, 38, 1],
    [760, lakeZ - 2, 4.5, 38, 1],
    [HILL.x + R * -0.5000, hz + R * 0.8660, 22.0, 36],
    [HILL.x + R * 0.0000, hz + R * 1.0000, 25.4, 36],
    [HILL.x + R * 0.5000, hz + R * 0.8660, 28.8, 36],
    [HILL.x + R * 0.8660, hz + R * 0.5000, 32.2, 36],
    [HILL.x + R * 1.0000, hz + R * 0.0000, 35.6, 36],
    [HILL.x + R * 0.8660, hz + R * -0.5000, 39.0, 36],
    [HILL.x + R * 0.5000, hz + R * -0.8660, 42.4, 36],
    [HILL.x + R * 0.0000, hz + R * -1.0000, 45.8, 36],
    [950, -118, 40, CRUISE],
    [800, -72, 32, CRUISE],
    [650, -45, 22, 32],
    [540, -28, 13, 20],
    [478, -18, 8, 10],
    [LAND.x, LAND.z, HOVER_AGL, 3],
  ].map(([x, z, agl, vcap, lock = 0]) => ({ p: [x, surfaceAt(x, z) + agl + g, z], agl, vcap, lock }));
  wp[0].p[1] = PAD[1] + g + HOVER_AGL;
  wp[wp.length - 1].p[1] = heightAt(LAND.x, LAND.z) + g + HOVER_AGL;

  // dense spline, arc length, curvature
  const pts = catmull(wp.map((w) => w.p), 60);
  const caps = [];
  for (let i = 0; i < wp.length - 1; i++) for (let s = 0; s < 60; s++) caps.push(lerp(wp[i].vcap, wp[i + 1].vcap, s / 60));
  caps.push(wp[wp.length - 1].vcap);
  const S = [0];
  for (let i = 1; i < pts.length; i++) {
    S.push(S[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]));
  }
  // resample uniformly in arc length
  const ds = 0.5, total = S[S.length - 1];
  const M = Math.floor(total / ds) + 1;
  const P = [], cap = [];
  let j = 0;
  for (let k = 0; k < M; k++) {
    const s = Math.min(k * ds, total);
    while (j < S.length - 2 && S[j + 1] < s) j++;
    const u = (s - S[j]) / (S[j + 1] - S[j] || 1);
    P.push([0, 1, 2].map((c) => lerp(pts[j][c], pts[j + 1][c], u)));
    cap.push(lerp(caps[j], caps[j + 1], u));
  }
  P[M - 1] = wp[wp.length - 1].p.slice();
  // terrain following: ground + planned AGL, smoothed over ~50 m of track, never lower
  // than 75 % of the planned clearance, ends pinned to the hover points
  // (waypoints flagged "lock" are flown at exactly their AGL: the low pass over the lake)
  const aglAt = [], lockAt = [];
  for (let i = 0; i < wp.length - 1; i++) for (let k = 0; k < 60; k++) {
    aglAt.push(lerp(wp[i].agl, wp[i + 1].agl, k / 60));
    lockAt.push(Math.min(wp[i].lock, wp[i + 1].lock) + (wp[i].lock !== wp[i + 1].lock ? 0 : 0));
  }
  aglAt.push(wp[wp.length - 1].agl); lockAt.push(wp[wp.length - 1].lock);
  const agl = [], lockR = [];
  { let jj = 0; for (let k = 0; k < M; k++) { const sk = Math.min(k * ds, total); while (jj < S.length - 2 && S[jj + 1] < sk) jj++; const u = (sk - S[jj]) / (S[jj + 1] - S[jj] || 1); agl.push(lerp(aglAt[jj], aglAt[jj + 1], u)); lockR.push(lockAt[jj]); } }
  const lock = gaussSmooth(lockR, 60 / ds);
  const grd = P.map((p) => surfaceAt(p[0], p[2]));
  let y = P.map((_, i) => grd[i] + agl[i] + g);
  // a pilot flies smooth lines, not every bump: heavy smoothing, then keep clearance
  for (let pass = 0; pass < 4; pass++) {
    y = gaussSmooth(y, 75 / ds);
    y = y.map((v, i) => Math.max(v, grd[i] + 0.7 * agl[i] + g));
  }
  y = y.map((v, i) => lerp(v, grd[i] + agl[i] + g, lock[i]));
  y = gaussSmooth(y, 25 / ds);
  const yEnds = [P[0][1], P[M - 1][1]];
  for (let i = 0; i < M; i++) {
    const wA = clamp(1 - i * ds / 30, 0, 1), wB = clamp(1 - (M - 1 - i) * ds / 30, 0, 1);
    P[i][1] = lerp(lerp(y[i], yEnds[0], wA), yEnds[1], wB);
  }
  // curvature via three-point circle, smoothed
  let kap = P.map((_, i) => {
    if (i < 4 || i > M - 5) return 0;
    const a = P[i - 4], b = P[i], c = P[i + 4];
    const ab = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    const bc = Math.hypot(c[0] - b[0], c[1] - b[1], c[2] - b[2]);
    const ac = Math.hypot(c[0] - a[0], c[1] - a[1], c[2] - a[2]);
    const s = (ab + bc + ac) / 2;
    const area = Math.sqrt(Math.max(0, s * (s - ab) * (s - bc) * (s - ac)));
    return 4 * area / (ab * bc * ac + 1e-9);
  });
  kap = gaussSmooth(kap, 40);
  // speed profile: caps, turn limit, then forward (accel) and backward (brake) passes
  const v = kap.map((k, i) => Math.min(cap[i], Math.sqrt(A_LAT / Math.max(k, 1e-6))));
  v[0] = 0; v[M - 1] = 0;
  for (let i = 1; i < M; i++) v[i] = Math.min(v[i], Math.sqrt(v[i - 1] ** 2 + 2 * A_ACC * ds));
  for (let i = M - 2; i >= 0; i--) v[i] = Math.min(v[i], Math.sqrt(v[i + 1] ** 2 + 2 * A_DEC * ds));
  // a pilot would not surge and brake every few seconds: smooth the plan, re-apply limits
  for (let pass = 0; pass < 2; pass++) {
    const vs = gaussSmooth(v, 45 / ds);
    for (let i = 0; i < M; i++) v[i] = Math.min(vs[i], cap[i] * 1.02, Math.sqrt(A_LAT * 1.1 / Math.max(kap[i], 1e-6)));
    v[0] = 0; v[M - 1] = 0;
    for (let i = 1; i < M; i++) v[i] = Math.min(v[i], Math.sqrt(v[i - 1] ** 2 + 2 * A_ACC * ds));
    for (let i = M - 2; i >= 0; i--) v[i] = Math.min(v[i], Math.sqrt(v[i + 1] ** 2 + 2 * A_DEC * ds));
  }
  // time along the path
  const tPath = [0];
  const segLen = (i) => Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1], P[i][2] - P[i - 1][2]);
  for (let i = 1; i < M; i++) tPath.push(tPath[i - 1] + segLen(i) / Math.max(0.25, (v[i] + v[i - 1]) / 2));

  // assemble full timeline, sampled at DT
  const ground = [PAD[0], PAD[1] + g, PAD[2]];
  const hover0 = wp[0].p, hover1 = wp[wp.length - 1].p;
  const tClimb = TL.liftoff, dClimb = 4.2, dHover0 = 2.0;
  const tPath0 = tClimb + dClimb + dHover0;
  const tPath1 = tPath0 + tPath[M - 1];
  const dHover1 = 2.5, dDescend = 7.5;
  const tTouchRef = tPath1 + dHover1 + dDescend;
  const touch = [hover1[0], heightAt(hover1[0], hover1[2]) + g - 0.25, hover1[2]];
  const tEnd = tTouchRef + 18;
  const minjerk = (a, b, u) => { u = clamp(u, 0, 1); const s = u * u * u * (10 - 15 * u + 6 * u * u); return a.map((_, k) => lerp(a[k], b[k], s)); };

  const n = Math.ceil(tEnd / DT) + 1;
  const X = [], Y = [], Z = [];
  let pi = 0;
  for (let i = 0; i < n; i++) {
    const t = i * DT;
    let p;
    if (t < tClimb) p = ground;
    else if (t < tClimb + dClimb) p = minjerk(ground, hover0, (t - tClimb) / dClimb);
    else if (t < tPath0) p = hover0;
    else if (t < tPath1) {
      const tp = t - tPath0;
      while (pi < M - 2 && tPath[pi + 1] < tp) pi++;
      const u = (tp - tPath[pi]) / (tPath[pi + 1] - tPath[pi]);
      p = [0, 1, 2].map((c) => lerp(P[pi][c], P[pi + 1][c], u));
    } else if (t < tPath1 + dHover1) p = hover1;
    else p = minjerk(hover1, touch, (t - tPath1 - dHover1) / dDescend);
    X.push(p[0]); Y.push(p[1]); Z.push(p[2]);
  }
  // light smoothing (jerk limiting) everywhere except the pre-liftoff hold
  const sm = (a) => gaussSmooth(a, 0.3 / DT);
  const Xs = sm(X), Ys = sm(Y), Zs = sm(Z);
  const iLift = Math.round(tClimb / DT);
  for (let i = 0; i <= iLift + 30; i++) { Xs[i] = X[i]; Ys[i] = Y[i]; Zs[i] = Z[i]; }
  const d1 = (a) => a.map((_, i) => (a[Math.min(n - 1, i + 1)] - a[Math.max(0, i - 1)]) / (DT * (i === 0 || i === n - 1 ? 1 : 2)));
  const VX = d1(Xs), VY = d1(Ys), VZ = d1(Zs);
  const AX = gaussSmooth(d1(VX), 5), AY = gaussSmooth(d1(VY), 5), AZ = gaussSmooth(d1(VZ), 5);

  // heading: along the horizontal velocity when moving, held when hovering
  const yaw = [];
  let hold = 0;
  for (let i = 0; i < n; i++) {
    const sp = Math.hypot(VX[i], VZ[i]);
    let y = Math.atan2(-VZ[i], VX[i]);
    if (sp > 2.5) {
      while (y - hold > Math.PI) y -= 2 * Math.PI;
      while (y - hold < -Math.PI) y += 2 * Math.PI;
      const wgt = clamp((sp - 2.5) / 4, 0, 1);
      hold = lerp(hold, y, wgt);
    }
    yaw.push(hold);
  }
  const yawS = gaussSmooth(yaw, 0.5 / DT);
  const yawRate = gaussSmooth(d1(yawS), 10);

  function sample(t) {
    const f = clamp(t / DT, 0, n - 1.001);
    const i = Math.floor(f), u = f - i;
    const L = (a) => lerp(a[i], a[i + 1], u);
    return {
      p: [L(Xs), L(Ys), L(Zs)],
      v: [L(VX), L(VY), L(VZ)],
      a: [L(AX), L(AY), L(AZ)],
      yaw: L(yawS),
      yawRate: L(yawRate),
    };
  }
  return {
    sample, tEnd, tTouchRef, tPath0, tPath1, pathLength: total, waypoints: wp,
    path: P.filter((_, i) => i % 10 === 0),
    lakeZ, hillZ: hz,
  };
}
