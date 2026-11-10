// print terrain height at points: node tools/probe.mjs x,z x,z ...
import { loadWorld } from '../tests/util.mjs';
import { groundAt, waterAt } from '../js/sim/terrain.js';
loadWorld();
for (const a of process.argv.slice(2)) { const [x, z] = a.split(',').map(Number); console.log(a, groundAt(x, z).toFixed(1), waterAt(x, z) > groundAt(x, z) ? 'WATER' : ''); }
