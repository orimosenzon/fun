// בדיקה: מריץ חלוקה למספר ערכי alpha ו-k ומדפיס סטיות. גם בודק רציפות.
import { createRequire } from 'module';
import fs from 'fs';
const require = createRequire(import.meta.url);
globalThis.window = globalThis;
eval(fs.readFileSync(new URL('../data.js', import.meta.url), 'utf8'));
const P = require('../partition.js');
const G = P.prepare(window.DATA, { inhabitedOnly: !!process.env.INHAB });
const ks = process.argv[2] ? [+process.argv[2]] : [10, 12, 13];
const mu = process.argv[3] ? +process.argv[3] : 1;
let bad = 0;
for (const k of ks) for (const alpha of [0, 0.25, 0.5, 0.75, 0.9, 1]) {
  const t = Date.now();
  const { reg } = P.solve(G, { k, alpha, mu, seed: 1, restarts: 2 });
  const s = P.summary(G, reg, k);
  for (let j = 0; j < k; j++) if (P.components(G, reg, j).length !== 1) { bad++; console.log('  לא רציף', k, alpha, j); }
  console.log(`k=${k} a=${alpha} maxP=${(s.maxP*100).toFixed(0)}% maxA=${(s.maxA*100).toFixed(0)}% rmsP=${(s.rmsP*100).toFixed(0)}% rmsA=${(s.rmsA*100).toFixed(0)}% cut=${s.cutKm.toFixed(0)}km ${Date.now()-t}ms`);
}
if (bad) { console.log('FAIL', bad); process.exit(1); } else console.log('OK');
