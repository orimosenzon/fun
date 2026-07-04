// Living scenery: horse herds grazing the meadows and bird flocks riding the
// valley air. Everything is InstancedMesh; the per-frame cost is only matrix
// composition (~500 matrices), no geometry churn.
//
// Horses: small herds on flat grass. Each horse is 8 instanced parts (body,
// neck, head, tail, 4 legs) composed from a shared root matrix, with a tiny
// state machine: graze (head down) <-> amble toward a nearby point (legs
// swinging, body bobbing).
//
// Birds: flocks in loose V formation. A flock is a wandering center + heading;
// each bird is a body and two wings that hinge at the shoulder, alternating
// flapping bursts with glides.
import * as THREE from 'three';
import { terrainHeight, terrainNormal, WATER_Y, SPAWN } from './world.js';
import { mulberry32 } from './noise.js';

const RX = Math.PI / 2;

export function makeAnimals(scene) {
  const rand = mulberry32(808);
  const _n = new THREE.Vector3();

  function grassAt(x, z) {
    const h = terrainHeight(x, z);
    if (h < WATER_Y + 2 || h > 95) return null;
    terrainNormal(x, z, _n);
    return _n.y > 0.94 ? h : null;
  }

  // ---------- horses ----------
  const horses = [];
  const HERD_R = 16;
  for (let t = 0; t < 4000 && horses.length < 44; t++) {
    const cx = (rand() * 2 - 1) * 1500, cz = (rand() * 2 - 1) * 1500;
    if (grassAt(cx, cz) === null) continue;
    if (Math.hypot(cx - SPAWN.x, cz - SPAWN.z) < 80) continue;
    if (horses.some(h => Math.hypot(cx - h.hx, cz - h.hz) < 260)) continue;
    const count = 3 + (rand() * 4 | 0);
    for (let i = 0; i < count; i++) {
      const x = cx + (rand() * 2 - 1) * HERD_R * 0.7;
      const z = cz + (rand() * 2 - 1) * HERD_R * 0.7;
      if (grassAt(x, z) === null) continue;
      horses.push({
        x, z, hx: cx, hz: cz,
        heading: rand() * Math.PI * 2,
        state: 'graze', timer: 2 + rand() * 8,
        tx: x, tz: z,
        phase: rand() * 7, neck: 2.0,
        s: 0.85 + rand() * 0.22,
        seed: rand() * 10,
      });
    }
  }

  const NH = horses.length;
  const coats = [0x6e4526, 0x3d2b1c, 0x1e1b18, 0xd8d2c6, 0xb98d4f, 0x8a6844];
  const horseMat = () => new THREE.MeshLambertMaterial({ color: 0xffffff });

  function horsePart(geo, perHorse = 1) {
    const m = new THREE.InstancedMesh(geo, horseMat(), NH * perHorse);
    m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    m.frustumCulled = false;
    scene.add(m);
    return m;
  }
  const bodyGeo = new THREE.BoxGeometry(0.55, 0.62, 1.55);
  bodyGeo.translate(0, 1.05, 0);
  const neckGeo = new THREE.BoxGeometry(0.24, 0.7, 0.32);
  neckGeo.translate(0, 0.3, 0.05);                    // origin at the shoulder pivot
  const headGeo = new THREE.BoxGeometry(0.2, 0.24, 0.55);
  headGeo.translate(0, 0, 0.16);
  const tailGeo = new THREE.BoxGeometry(0.09, 0.55, 0.11);
  tailGeo.translate(0, -0.26, -0.02);
  const legGeo = new THREE.CylinderGeometry(0.055, 0.042, 0.98, 6);
  legGeo.translate(0, -0.49, 0);                      // origin at the hip

  const hBody = horsePart(bodyGeo), hNeck = horsePart(neckGeo), hHead = horsePart(headGeo);
  const hTail = horsePart(tailGeo), hLegs = horsePart(legGeo, 4);
  {
    const c = new THREE.Color(), dk = new THREE.Color();
    horses.forEach((h, i) => {
      c.setHex(coats[(h.seed * 0.6 | 0) % coats.length]);
      dk.copy(c).multiplyScalar(0.55);
      hBody.setColorAt(i, c); hNeck.setColorAt(i, c); hHead.setColorAt(i, c);
      hTail.setColorAt(i, dk);
      for (let l = 0; l < 4; l++) hLegs.setColorAt(i * 4 + l, dk);
    });
  }

  const HIPS = [[0.17, 0.98, 0.58], [-0.17, 0.98, 0.58], [0.17, 0.98, -0.55], [-0.17, 0.98, -0.55]];
  const LEG_PHASE = [0, Math.PI, Math.PI, 0];

  // ---------- birds ----------
  const flocks = [];
  for (let f = 0; f < 9; f++) {
    let cx = 0, cz = 0, ok = false;
    for (let t = 0; t < 60 && !ok; t++) {
      cx = (rand() * 2 - 1) * 1400; cz = (rand() * 2 - 1) * 1400;
      ok = terrainHeight(cx, cz) < 180;
    }
    const overWater = terrainHeight(cx, cz) < WATER_Y + 3;
    const count = 5 + (rand() * 4 | 0);
    const offsets = [];
    for (let i = 0; i < count; i++) {
      const rank = (i + 1) >> 1, side = i % 2 ? 1 : -1;
      offsets.push(new THREE.Vector3(
        side * rank * 1.7 + (rand() - 0.5), (rand() - 0.5) * 0.8, -rank * 1.5 + (rand() - 0.5)));
    }
    flocks.push({
      cx, cz, y: 0, alt: 35 + rand() * 55,
      heading: rand() * Math.PI * 2,
      turnBias: (rand() - 0.5) * 0.22,
      speed: 9 + rand() * 5,
      seed: rand() * 20,
      color: overWater ? 0xdde2e6 : 0x2b2620,
      offsets,
    });
    flocks[f].y = Math.max(terrainHeight(cx, cz), WATER_Y) + flocks[f].alt;
  }
  const NB = flocks.reduce((s, f) => s + f.offsets.length, 0);

  const birdBodyGeo = new THREE.ConeGeometry(0.075, 0.5, 5);
  birdBodyGeo.rotateX(RX);                            // point the beak along +Z
  function wingGeo(side) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute([
      0, 0, 0.12, side * 0.62, 0.02, -0.06, 0, 0, -0.2], 3));
    g.computeVertexNormals();
    return g;
  }
  function birdPart(geo) {
    const m = new THREE.InstancedMesh(
      geo, new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide }), NB);
    m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    m.frustumCulled = false;
    scene.add(m);
    return m;
  }
  const bBody = birdPart(birdBodyGeo), bWingL = birdPart(wingGeo(1)), bWingR = birdPart(wingGeo(-1));
  {
    const c = new THREE.Color();
    let i = 0;
    for (const f of flocks) for (const _ of f.offsets) {
      c.setHex(f.color);
      bBody.setColorAt(i, c); bWingL.setColorAt(i, c); bWingR.setColorAt(i, c);
      i++;
    }
  }

  // ---------- per-frame ----------
  const M = new THREE.Matrix4(), P = new THREE.Matrix4(), R = new THREE.Matrix4();
  const Q = new THREE.Quaternion(), V = new THREE.Vector3(), S = new THREE.Vector3();
  const Y_AXIS = new THREE.Vector3(0, 1, 0);
  let time = 0;

  function update(dt, playerPos) {
    time += dt;

    // --- horses ---
    for (let i = 0; i < NH; i++) {
      const h = horses[i];
      h.timer -= dt;
      let walking = h.state === 'walk';
      if (h.timer <= 0) {
        if (walking) { h.state = 'graze'; h.timer = 4 + rand() * 9; walking = false; }
        else {
          const tx = h.hx + (rand() * 2 - 1) * HERD_R, tz = h.hz + (rand() * 2 - 1) * HERD_R;
          if (grassAt(tx, tz) !== null) { h.state = 'walk'; h.tx = tx; h.tz = tz; h.timer = 20; walking = true; }
          else h.timer = 1;
        }
      }
      if (walking) {
        const dx = h.tx - h.x, dz = h.tz - h.z, d = Math.hypot(dx, dz);
        if (d < 0.8) { h.state = 'graze'; h.timer = 4 + rand() * 9; walking = false; }
        else {
          const want = Math.atan2(dx, dz);
          let dh = want - h.heading;
          while (dh > Math.PI) dh -= 2 * Math.PI;
          while (dh < -Math.PI) dh += 2 * Math.PI;
          h.heading += Math.max(-1.6 * dt, Math.min(1.6 * dt, dh));
          const sp = 1.6 * h.s;
          h.x += Math.sin(h.heading) * sp * dt;
          h.z += Math.cos(h.heading) * sp * dt;
          h.phase += dt * 5.2;
        }
      }
      // neck: down to the grass when grazing, up when moving
      const neckWant = walking ? 0.55 : 2.15 + 0.1 * Math.sin(time * 0.5 + h.seed);
      h.neck += (neckWant - h.neck) * Math.min(1, dt * 2.5);

      // skip matrix work for far-away herds (they still render, frozen)
      if (Math.hypot(h.x - playerPos.x, h.z - playerPos.z) > 700) continue;

      const y = terrainHeight(h.x, h.z) + (walking ? Math.abs(Math.sin(h.phase)) * 0.05 : 0);
      Q.setFromAxisAngle(Y_AXIS, h.heading);
      M.compose(V.set(h.x, y, h.z), Q, S.set(h.s, h.s, h.s));

      hBody.setMatrixAt(i, P.copy(M).multiply(R.identity()));
      // neck pivots at the shoulder; head rides the neck tip, leveled back
      P.copy(M).multiply(R.makeTranslation(0, 1.25, 0.62)).multiply(R2rx(h.neck));
      hNeck.setMatrixAt(i, P);
      P.multiply(R.makeTranslation(0, 0.64, 0.1)).multiply(R2rx(-h.neck * 0.55 + 0.5));
      hHead.setMatrixAt(i, P);
      P.copy(M).multiply(R.makeTranslation(0, 1.32, -0.76)).multiply(R2rx(-0.45 + 0.12 * Math.sin(time * 1.7 + h.seed)));
      hTail.setMatrixAt(i, P);
      for (let l = 0; l < 4; l++) {
        const swing = walking ? Math.sin(h.phase + LEG_PHASE[l]) * 0.42 : 0;
        P.copy(M).multiply(R.makeTranslation(HIPS[l][0], HIPS[l][1], HIPS[l][2])).multiply(R2rx(swing));
        hLegs.setMatrixAt(i * 4 + l, P);
      }
    }
    hBody.instanceMatrix.needsUpdate = true;
    hNeck.instanceMatrix.needsUpdate = true;
    hHead.instanceMatrix.needsUpdate = true;
    hTail.instanceMatrix.needsUpdate = true;
    hLegs.instanceMatrix.needsUpdate = true;

    // --- birds ---
    let bi = 0;
    for (const f of flocks) {
      const wob = f.turnBias + 0.3 * Math.sin(time * 0.11 + f.seed);
      f.heading += wob * dt;
      // steer home if drifting off the world
      if (Math.hypot(f.cx, f.cz) > 1600) {
        const back = Math.atan2(-f.cx, -f.cz);
        let dh = back - f.heading;
        while (dh > Math.PI) dh -= 2 * Math.PI;
        while (dh < -Math.PI) dh += 2 * Math.PI;
        f.heading += Math.max(-0.5 * dt, Math.min(0.5 * dt, dh));
      }
      f.cx += Math.sin(f.heading) * f.speed * dt;
      f.cz += Math.cos(f.heading) * f.speed * dt;
      const yWant = Math.max(terrainHeight(f.cx, f.cz), WATER_Y) + f.alt;
      f.y += (yWant - f.y) * Math.min(1, dt * 0.6);

      Q.setFromAxisAngle(Y_AXIS, f.heading);
      // flapping comes in bursts; between bursts the flock glides
      const burst = 0.2 + 0.8 * Math.max(0, Math.sin(time * 0.4 + f.seed));
      for (let k = 0; k < f.offsets.length; k++, bi++) {
        V.copy(f.offsets[k]).applyQuaternion(Q);
        V.x += f.cx; V.z += f.cz;
        V.y += f.y + 0.5 * Math.sin(time * 1.3 + k * 1.9 + f.seed);
        M.compose(V, Q, S.set(1, 1, 1));
        bBody.setMatrixAt(bi, M);
        const flap = Math.sin(time * 9 + k * 1.7 + f.seed) * 0.75 * burst + 0.12;
        bWingL.setMatrixAt(bi, P.copy(M).multiply(R2rz(flap)));
        bWingR.setMatrixAt(bi, P.copy(M).multiply(R2rz(-flap)));
      }
    }
    bBody.instanceMatrix.needsUpdate = true;
    bWingL.instanceMatrix.needsUpdate = true;
    bWingR.instanceMatrix.needsUpdate = true;
  }

  // tiny rotation-matrix helpers (avoid allocating in the loop)
  const _rx = new THREE.Matrix4(), _rz = new THREE.Matrix4();
  function R2rx(a) { return _rx.makeRotationX(a); }
  function R2rz(a) { return _rz.makeRotationZ(a); }

  // horses/flocks exposed for the ?pose=horses / ?pose=birds debug cameras
  return { update, horses, flocks };
}
