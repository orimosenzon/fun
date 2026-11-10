import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { initTerrain } from '../js/sim/terrain.js';
export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export function loadWorld() {
  const meta = JSON.parse(fs.readFileSync(path.join(ROOT, 'assets/meta.json'), 'utf8'));
  const buf = fs.readFileSync(path.join(ROOT, 'assets/terrain.bin'));
  initTerrain(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), meta);
  return meta;
}
export const deg = (r) => r * 180 / Math.PI;
