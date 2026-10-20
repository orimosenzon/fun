// החומרים של אופטימוס 3, לפי ההדמיות שדלפו מאפליקציית טסלה (ספטמבר 2026):
// גוף "זהב שמפניה" מתכתי ומעוגל, ראש שחור מבריק עם מגן פנים מראתי, חליפה שחורה מהמותן עד הירכיים,
// ו"מגפיים" שחורים מט מהברך ומטה. כפות ידיים שחורות.
import * as THREE from 'three';

function fabricTexture(size = 256) {
  // אריג סרוג עדין: מפת חספוס ומפת נורמלים פרוצדורליות
  const c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d');
  const img = g.createImageData(size, size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const k = (Math.sin(x * 0.9) * 0.5 + 0.5) * (Math.sin(y * 1.6 + (x % 4 < 2 ? 0 : 1.6)) * 0.5 + 0.5);
    const v = 150 + k * 70 + (Math.random() - 0.5) * 20;
    const i = (y * size + x) * 4;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v; img.data[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(6, 6);
  return t;
}

let cache = null;
export function materials() {
  if (cache) return cache;
  const fab = typeof document !== 'undefined' ? fabricTexture() : null;
  cache = {
    gold: new THREE.MeshPhysicalMaterial({ color: 0xcdb892, metalness: 0.88, roughness: 0.3, clearcoat: 0.35, clearcoatRoughness: 0.25 }),
    goldDark: new THREE.MeshPhysicalMaterial({ color: 0xa8946d, metalness: 0.85, roughness: 0.38 }),
    shell: new THREE.MeshPhysicalMaterial({ color: 0x0d0d0e, metalness: 0.35, roughness: 0.42, clearcoat: 0.6, clearcoatRoughness: 0.3 }),
    boot: new THREE.MeshPhysicalMaterial({ color: 0x131314, metalness: 0.15, roughness: 0.55, clearcoat: 0.25, clearcoatRoughness: 0.5 }),
    fabric: new THREE.MeshStandardMaterial({ color: 0x18181a, roughness: 0.92, metalness: 0.0, roughnessMap: fab, bumpMap: fab, bumpScale: 0.6 }),
    helmet: new THREE.MeshPhysicalMaterial({ color: 0x0b0b0d, metalness: 0.3, roughness: 0.28, clearcoat: 1.0, clearcoatRoughness: 0.1 }),
    visor: new THREE.MeshPhysicalMaterial({ color: 0x1a1d24, metalness: 1.0, roughness: 0.04, clearcoat: 1.0, clearcoatRoughness: 0.02, envMapIntensity: 2.2 }),
    rim: new THREE.MeshStandardMaterial({ color: 0xf2f4f7, metalness: 1.0, roughness: 0.12, envMapIntensity: 2.0 }),
    hand: new THREE.MeshPhysicalMaterial({ color: 0x141416, metalness: 0.2, roughness: 0.5, clearcoat: 0.3 }),
    pad: new THREE.MeshStandardMaterial({ color: 0x2a2b2e, metalness: 0.0, roughness: 0.85 }),
    knuckle: new THREE.MeshStandardMaterial({ color: 0x3a3b3f, metalness: 0.6, roughness: 0.35 }),
    joint: new THREE.MeshStandardMaterial({ color: 0x232325, metalness: 0.6, roughness: 0.4 }),
  };
  return cache;
}
