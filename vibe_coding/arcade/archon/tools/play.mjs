// node tools/play.mjs "<script>" <seconds> <shot every N seconds> [name] [model] [disk]
// Runs the game with scripted input and saves screenshots to reference/shots/.
// Script: comma separated "time:ACTION", time in seconds. Actions:
//   +KEY / -KEY press / release a key (names from KEYS), KEY alone = tap for 0.15 s
//   J1:UDLRF / J2:UDLRF set a joystick's state (empty = released), e.g. 12:J2:RF
import fs from 'node:fs';
import { startGame, shot, KEYS, JOY } from './node64.mjs';

const [script = '', secs = '30', every = '5', name = 'run', model = 'ntsc', disk = 'data/archon.d64'] = process.argv.slice(2);
const m = startGame({ model, disk });
const fps = m.model.hz / (m.model.lines * m.model.cpl);
const events = [];
for (const item of script.split(',').filter(Boolean)) {
  const [t, a, b] = item.split(':');
  const at = Math.round(parseFloat(t) * fps);
  if (a === 'J1' || a === 'J2') {
    let bits = 0;
    for (const ch of b || '') bits |= { U: JOY.UP, D: JOY.DOWN, L: JOY.LEFT, R: JOY.RIGHT, F: JOY.FIRE }[ch];
    events.push({ at, fn: () => { m.joy[a === 'J1' ? 0 : 1] = bits; } });
  } else if (a[0] === '+' || a[0] === '-') {
    events.push({ at, fn: () => m.setKey(KEYS[a.slice(1)], a[0] === '+') });
  } else {
    events.push({ at, fn: () => m.setKey(KEYS[a], true) }, { at: at + Math.round(0.15 * fps), fn: () => m.setKey(KEYS[a], false) });
  }
}
const dir = new URL('../reference/shots/', import.meta.url).pathname;
fs.mkdirSync(dir, { recursive: true });
const total = Math.round(parseFloat(secs) * fps), step = Math.round(parseFloat(every) * fps);
const t0 = Date.now();
for (let f = 0; f <= total; f++) {
  for (const e of events) if (e.at === f) e.fn();
  m.runFrame();
  if (m.cpu.jammed) { console.log('CPU jammed at $' + m.cpu.pc.toString(16), 'frame', f); break; }
  if (f % step === 0 && f) { const file = `${dir}${name}_${String(Math.round(f / fps)).padStart(4, '0')}.png`; shot(m, file); }
}
console.log('ran', total, 'frames in', Date.now() - t0, 'ms; pc $' + m.cpu.pc.toString(16), 'port', m.port.toString(16));
console.log(m.screenText().filter((l) => l.trim()).slice(0, 12).join('\n'));
