// Glowing course rings (HDR emissive, so they bloom) and the precision-landing target.

import * as THREE from 'three';
import { RING_R, LANDING } from '../course.js';
import { heightAt } from '../terrain.js';

export function buildRings(scene, rings) {
  const geo = new THREE.TorusGeometry(RING_R + 0.55, 0.42, 16, 72);
  const items = rings.map((r, i) => {
    const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false, fog: false });
    const m = new THREE.Mesh(geo, mat);
    m.position.set(...r.p);
    // torus lies in its local x-y plane; face its axis (+z) along the direction of travel
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(...r.n));
    // four small beacons on the rim
    for (let k = 0; k < 4; k++) {
      const b = new THREE.Mesh(new THREE.SphereGeometry(0.55, 12, 8), mat);
      const a = k * Math.PI / 2 + Math.PI / 4;
      b.position.set(Math.cos(a) * (RING_R + 0.55), Math.sin(a) * (RING_R + 0.55), 0);
      m.add(b);
    }
    // number plate
    const c = document.createElement('canvas'); c.width = 128; c.height = 128;
    const g = c.getContext('2d');
    g.fillStyle = '#fff'; g.font = 'bold 84px Heebo, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText(String(i + 1), 64, 70);
    const tex = new THREE.CanvasTexture(c);
    const plate = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, color: 0xffffff, transparent: true, depthWrite: false }));
    plate.scale.set(3.2, 3.2, 1);
    plate.position.set(0, RING_R + 3.2, 0);
    m.add(plate);
    scene.add(m);
    return { mesh: m, mat, plate };
  });

  // landing target: painted ring on the meadow + marker flags
  const lt = new THREE.Group();
  const y = heightAt(LANDING.x, LANDING.z);
  lt.position.set(LANDING.x, y + 0.06, LANDING.z);
  const tMat = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false, transparent: true, opacity: 0.9, depthWrite: false });
  const disc = new THREE.Mesh(new THREE.RingGeometry(LANDING.r - 0.35, LANDING.r, 64).rotateX(-Math.PI / 2), tMat);
  const inner = new THREE.Mesh(new THREE.RingGeometry(1.6, 1.9, 48).rotateX(-Math.PI / 2), tMat);
  lt.add(disc, inner);
  const poleM = new THREE.MeshStandardMaterial({ color: 0xdddddd, roughness: 0.6 });
  const flagM = new THREE.MeshStandardMaterial({ color: 0xe8590c, roughness: 0.7, side: THREE.DoubleSide, emissive: 0x551800 });
  for (let k = 0; k < 6; k++) {
    const a = k / 6 * Math.PI * 2;
    const x = Math.cos(a) * (LANDING.r + 1.2), z = Math.sin(a) * (LANDING.r + 1.2);
    const gy = heightAt(LANDING.x + x, LANDING.z + z) - y;
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.03, 2.2, 6), poleM);
    pole.position.set(x, gy + 1.1, z);
    const flag = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 0.45), flagM);
    flag.position.set(x + 0.35, gy + 1.95, z);
    pole.castShadow = flag.castShadow = true;
    lt.add(pole, flag);
  }
  scene.add(lt);

  const cNext = new THREE.Color(5.0, 1.9, 0.35), cLater = new THREE.Color(0.9, 0.42, 0.12), cDone = new THREE.Color(0.25, 1.2, 0.5);
  return {
    update(next, t, finished) {
      items.forEach((it, i) => {
        const pulse = 0.75 + 0.25 * Math.sin(t * 5);
        if (i < next) { it.mesh.visible = false; }
        else if (i === next) { it.mat.color.copy(cNext).multiplyScalar(pulse); it.mesh.visible = true; }
        else { it.mat.color.copy(cLater); it.mesh.visible = true; }
        it.plate.material.opacity = i === next ? 1 : 0.45;
      });
      const landingNext = next >= items.length && !finished;
      tMat.color.setRGB(landingNext ? 4.5 : 0.8, landingNext ? 1.7 : 0.4, landingNext ? 0.3 : 0.1).multiplyScalar(landingNext ? 0.75 + 0.25 * Math.sin(t * 5) : 1);
    },
    reset() { items.forEach((it) => { it.mesh.visible = true; }); },
  };
}
