// The jet bike mesh, built from primitives. Body frame matches physics.js:
// X = right, Y = up, Z = forward. Exhaust flames live in jetfx.js.
import * as THREE from 'three';

export function buildBike() {
  const g = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0xb3352c, metalness: 0.55, roughness: 0.35 });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x23262b, metalness: 0.7, roughness: 0.5 });
  const chromeMat = new THREE.MeshStandardMaterial({ color: 0x9aa4ad, metalness: 0.9, roughness: 0.25 });

  const hull = new THREE.Mesh(new THREE.CapsuleGeometry(0.38, 1.9, 6, 14), bodyMat);
  hull.rotation.x = Math.PI / 2;
  g.add(hull);

  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.3, 0.7, 12), bodyMat);
  nose.rotation.x = Math.PI / 2;
  nose.position.z = 1.6;
  g.add(nose);

  const seat = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.22, 1.0), darkMat);
  seat.position.set(0, 0.38, -0.5);
  g.add(seat);

  const shield = new THREE.Mesh(
    new THREE.BoxGeometry(0.5, 0.42, 0.06),
    new THREE.MeshStandardMaterial({ color: 0x9fd8ff, transparent: true, opacity: 0.4, metalness: 0.2, roughness: 0.1 })
  );
  shield.position.set(0, 0.5, 0.78);
  shield.rotation.x = -0.5;
  g.add(shield);

  const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.85), chromeMat);
  bar.rotation.z = Math.PI / 2;
  bar.position.set(0, 0.42, 0.62);
  g.add(bar);

  for (const s of [-1, 1]) {
    const pod = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.3, 1.0), darkMat);
    pod.position.set(s * 0.6, -0.12, 0.15);
    g.add(pod);
    const noz = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.2, 0.35, 12), chromeMat);
    noz.position.set(s * 0.75, -0.32, 0);
    g.add(noz);
    const skid = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.06, 1.5), chromeMat);
    skid.position.set(s * 0.38, -0.52, -0.1);
    g.add(skid);
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.25, 0.06), darkMat);
    leg.position.set(s * 0.38, -0.38, 0.3);
    g.add(leg);
    const leg2 = leg.clone();
    leg2.position.z = -0.5;
    g.add(leg2);
  }

  const nozC = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.22, 0.35, 12), chromeMat);
  nozC.position.set(0, -0.42, 0);
  g.add(nozC);

  const rearNoz = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.24, 0.5, 14), darkMat);
  rearNoz.rotation.x = Math.PI / 2;
  rearNoz.position.set(0, 0, -1.5);
  g.add(rearNoz);

  return { group: g };
}
