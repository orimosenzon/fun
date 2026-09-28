// The film's shot list. Every camera is a function of time that may look ahead in the
// recorded flight (it is all precomputed), the way a real camera crew would pre-plan.

import * as THREE from 'three';
import { heightAt, surfaceAt, WATER_Y, LAND, HILL, valleyZ, LAKE } from '../terrain.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const P = (s) => V(s.p[0], s.p[1], s.p[2]);
const Q = (s) => new THREE.Quaternion(s.q[1], s.q[2], s.q[3], s.q[0]);

// time-smoothed bike position (camera operators don't follow micro jitter)
function smoothPos(rec, t, win = 0.4, n = 9) {
  const acc = V(0, 0, 0);
  let ws = 0;
  for (let k = 0; k < n; k++) {
    const u = (k / (n - 1) - 0.5) * win;
    const w = Math.exp(-8 * (u / win) ** 2);
    acc.addScaledVector(P(rec.at(t + u)), w); ws += w;
  }
  return acc.multiplyScalar(1 / ws);
}
function handheld(t, amp = 1, seed = 0) {
  const n = (f, p) => Math.sin(t * f + p + seed) * 0.6 + Math.sin(t * f * 2.13 + p * 1.7 + seed) * 0.3 + Math.sin(t * f * 4.7 + p + seed * 2) * 0.1;
  return V(n(0.9, 0.3), n(1.1, 1.9), n(0.7, 4.1)).multiplyScalar(0.05 * amp);
}
const ease = (u) => u * u * (3 - 2 * u);
function firstTime(rec, from, pred, step = 0.05) {
  for (let t = from; t < rec.duration; t += step) if (pred(rec.at(t), t)) return t;
  return from;
}

export function makeShots(rec) {
  const ev = rec.events;
  const lift = ev.liftoff;
  const tLake = firstTime(rec, 20, (s) => s.p[0] > 600);
  const tHill = firstTime(rec, tLake, (s) => s.p[0] > HILL.x - 250);
  const tTurnMid = firstTime(rec, tHill, (s) => s.p[2] < valleyZ(HILL.x) + HILL.dz && s.p[0] > HILL.x + 150);
  const tBack = firstTime(rec, tTurnMid, (s) => s.p[0] < HILL.x - 60);
  const tLandHover = firstTime(rec, tBack + 5, (s) => Math.hypot(s.p[0] - LAND.x, s.p[2] - LAND.z) < 60);
  const tEnd = ev.shutdown + 9;
  const hz = valleyZ(HILL.x) + HILL.dz;
  const lakeZ = valleyZ(LAKE.x) + LAKE.dz;

  const shots = [
    { name: 'pad-wide', t0: 0, t1: 6.5, cam(t) {
      const s = rec.at(t), b = P(s);
      const u = ease(t / 6.5);
      const ang = 2.35 - 0.35 * u, r = 9.5 - 2.5 * u;
      const pos = V(b.x + Math.cos(ang) * r, 1.3 + 0.2 * u, b.z + Math.sin(ang) * r).add(handheld(t, 0.6));
      return { pos, look: b.clone().add(V(0.1, 0.35, 0)), fov: 38 };
    } },
    { name: 'nozzle-close', t0: 6.5, t1: lift - 0.2, cam(t) {
      const s = rec.at(t), b = P(s);
      const u = (t - 6.5) / (lift - 6.7);
      const pos = V(b.x + 2.3 - 0.5 * u, 0.45, b.z - 1.9 + 0.2 * u).add(handheld(t, 0.4, 3));
      return { pos, look: b.clone().add(V(0.4, -0.1, -0.2)), fov: 42 };
    } },
    { name: 'liftoff-low', t0: lift - 0.2, t1: 18.5, cam(t) {
      const b = smoothPos(rec, t, 0.3);
      const pos = V(-6, 0.9, 15).add(handheld(t, 1.0, 5));
      return { pos, look: b.clone().add(V(0, 0.2, 0)), fov: 34 - 6 * ease(Math.min(1, (t - lift) / 8)) };
    } },
    { name: 'chase', t0: 18.5, t1: 28.5, cam(t) {
      const b = smoothPos(rec, t, 0.5);
      const s = rec.at(t);
      const v = V(...s.v); const sp = v.length();
      const back = sp > 1 ? v.clone().setY(0).normalize() : V(1, 0, 0);
      const side = V(-back.z, 0, back.x);
      const dist = 9 + 0.12 * sp;
      const pos = b.clone().addScaledVector(back, -dist).addScaledVector(side, -3.5).add(V(0, 2.4, 0)).add(handheld(t, 0.8, 7));
      return { pos, look: b.clone().addScaledVector(back, 6).add(V(0, 0.3, 0)), fov: 46 };
    } },
    { name: 'heli-track', t0: 28.5, t1: tLake - 3.5, cam(t) {
      const b = smoothPos(rec, t, 1.0, 11);
      const s = rec.at(t);
      const v = V(...s.v).setY(0).normalize();
      const side = V(-v.z, 0, v.x);
      const u = (t - 28.5) / Math.max(1, tLake - 32);
      const pos = b.clone().addScaledVector(side, 34).addScaledVector(v, 22 - 30 * u).add(V(0, 10, 0)).add(handheld(t, 1.2, 9));
      return { pos, look: b.clone().addScaledVector(v, 1.5), fov: 24 };
    } },
    { name: 'lake-flyby', t0: tLake - 3.5, t1: tLake + 4.5, cam(t) {
      const cp = V(LAKE.x + 25, WATER_Y + 1.1, lakeZ + 22);
      const b = smoothPos(rec, t, 0.25);
      return { pos: cp.clone().add(handheld(t, 0.7, 11)), look: b, fov: 30 };
    } },
    { name: 'onboard', t0: tLake + 4.5, t1: tHill + 5, onboard: true, cam(t) {
      const s = rec.at(t);
      const q = Q(s), b = P(s);
      const vib = V(Math.sin(t * 91) * 0.004, Math.sin(t * 77 + 1) * 0.004, Math.sin(t * 83 + 2) * 0.003);
      // action camera on the handlebar clamp, looking back at the rider
      const pos = V(0.97, 0.78, 0.22).add(vib).applyQuaternion(q).add(b);
      const look = V(-1.2, 0.55, -0.05).applyQuaternion(q).add(b);
      const up = V(0, 1, 0).applyQuaternion(q);
      return { pos, look, up, fov: 70, near: 0.05 };
    } },
    { name: 'turn-heli', t0: tHill + 5, t1: tBack + 3, cam(t) {
      // helicopter outside the turn, keeping the bike between camera and the rock
      const hc = V(HILL.x, 0, hz);
      const b = smoothPos(rec, t, 0.8, 11);
      const rad = V(b.x - hc.x, 0, b.z - hc.z);
      const r = rad.length(); rad.normalize();
      const tang = V(-rad.z, 0, rad.x);
      const pos = hc.clone().addScaledVector(rad, r + 26).addScaledVector(tang, -10);
      pos.y = b.y + 3.5;
      return { pos: pos.add(handheld(t, 1.4, 13)), look: b.clone().add(V(0, -0.5, 0)), fov: 30 };
    } },
    { name: 'front-chase', t0: tBack + 3, t1: tBack + 11, cam(t) {
      const b = smoothPos(rec, t, 0.6);
      const s = rec.at(t);
      const v = V(...s.v).setY(0).normalize();
      const side = V(-v.z, 0, v.x);
      const pos = b.clone().addScaledVector(v, 13).addScaledVector(side, 3.5).add(V(0, 1.2, 0)).add(handheld(t, 0.8, 17));
      return { pos, look: b.clone().add(V(0, 0.4, 0)), fov: 40 };
    } },
    { name: 'shore-flare', t0: tBack + 11, t1: tLandHover + 2, cam(t) {
      const cp = V(575, 0, 26);
      cp.y = surfaceAt(cp.x, cp.z) + 3.5;
      const b = smoothPos(rec, t, 0.5);
      const d = b.distanceTo(cp);
      // operator zooms to keep the bike about the same size in frame
      const fov = THREE.MathUtils.clamp(2 * Math.atan(9 / d) * 57.3, 9, 45);
      return { pos: cp.add(handheld(t, 0.5, 19)), look: b, fov };
    } },
    { name: 'landing', t0: tLandHover + 2, t1: ev.shutdown + 1.5, cam(t) {
      const cp = V(LAND.x - 7.5, 0, LAND.z + 9.5);
      cp.y = heightAt(cp.x, cp.z) + 1.0;
      const b = smoothPos(rec, t, 0.4);
      return { pos: cp.add(handheld(t, 0.7, 23)), look: b.clone().add(V(0, 0.1, 0)), fov: 38 };
    } },
    { name: 'crane-out', t0: ev.shutdown + 1.5, t1: tEnd, cam(t) {
      const b = P(rec.at(t));
      const u = ease(Math.min(1, (t - ev.shutdown - 1.5) / (tEnd - ev.shutdown - 1.5)));
      const ang = -0.9 + 0.55 * u;
      const r = 6 + 16 * u;
      const pos = V(b.x + Math.cos(ang) * r, b.y + 0.5 + 9 * u, b.z + Math.sin(ang) * r);
      return { pos, look: b.clone().add(V(-8 * u, 0.2 - 2 * u, 0)), fov: 40 };
    } },
  ];
  // fixed camera spots: the world keeps these clear of trees
  const clearings = [[-6, 15], [LAKE.x + 25, lakeZ + 22], [575, 26], [LAND.x - 7.5, LAND.z + 9.5]];
  return {
    shots, tEnd, clearings,
    marks: { lift, tLake, tHill, tTurnMid, tBack, tLandHover },
    at(t) {
      let sh = shots[shots.length - 1];
      for (const s of shots) if (t >= s.t0 && t < s.t1) { sh = s; break; }
      return { shot: sh, ...sh.cam(Math.min(t, sh.t1)) };
    },
  };
}
