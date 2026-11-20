// node tools/demo_shots.mjs: lets the computer play a whole beginner's game against itself (the
// "computer demo" option, fire held to hurry it) and saves a late map and the closing screens to guide/.
import fs from 'node:fs';
import { Session } from './lab.mjs';
import { shot } from './node64.mjs';
const out = new URL('../guide/', import.meta.url).pathname;
const s = new Session('demo', { quiet: true });
const snap = (name) => { shot(s.m, `${out}${name}.png`, 2); console.log(name.padEnd(14), s.t.toFixed(0), s.text().slice(0, 300)); };
s.wait(3).joy(1, 'F', 0.3).wait(13);
for (let i = 0; i < 4; i++) s.key('F5', 0.15, 0.6);
s.key('F7').wait(2);
let last = '', n = 0, late = false;
s.hold(1, 'F');
for (let f = 0; f < 60 * 60 * 40 && n < 6; f++) {
  s.frame();
  if (f % 30) continue;
  const t = s.text(), head = t.split(' | ')[0];
  // Fire hurries the computer players and answers "press buttons to go on". Near the end it is
  // left alone so the closing screens stay up long enough to be photographed.
  if (!late && f % 60 === 0) s.hold(1, (f / 60) % 2 ? 'F' : '');
  if (head === last) continue;
  last = head; console.log(s.t.toFixed(0), head);
  if (/^PRODUCTION #6/.test(t)) { s.wait(2); snap('demo_production'); }
  if (/AUCTION #6/.test(t) && /ENERGY/.test(t)) { late = true; s.hold(1, ''); }
  if (late && !/STATUS #6|AUCTION #6/.test(t) && t.length > 20) { s.wait(4); snap('demo_end' + (++n)); s.joy(1, 'F', 0.3); }
}
