// print a range of the study listing compactly: node tools/show.mjs B7A0 B900
import fs from 'node:fs';
const [a0, a1] = process.argv.slice(2, 4).map((x) => parseInt(x, 16)), file = process.argv[4] || 'game';
let on = false; const out = [];
for (const l of fs.readFileSync(new URL(`../re/${file}.lst`, import.meta.url), 'utf8').split('\n')) {
  const m = /^([0-9A-F]{4})  (.*?)\s*;(.*)$/.exec(l);
  if (m) { const a = parseInt(m[1], 16); on = a >= a0 && a <= a1; if (on) out.push(m[1] + ' ' + m[2].trim() + (m[3].trim() === '-' ? '  ;-' : '')); }
  else if (on || /^(sub|L)_/.test(l)) { const k = /^(sub|L)_([0-9A-F]{4}):\s*; from (.*)$/.exec(l); if (k) { const a = parseInt(k[2], 16); if (a >= a0 && a <= a1) out.push(`${k[1] === 'sub' ? '\n' : ''}${k[1]}_${k[2]}: <${k[3]}`); } else if (l.startsWith('; ----') && on) out.push('---'); }
}
console.log(out.join('\n'));
