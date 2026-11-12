// Save states: a game loaded into a fresh machine goes on exactly like the original,
// frame by frame and cycle by cycle, including in the middle of a somersault and a mawashi-geri.
import { newGame, startGame } from './lab.mjs';
import { installMawashi } from '../src/ext/mawashi.js';
import { installFlip } from '../src/ext/flip.js';
import { saveState, loadState } from '../src/savestate.js';
let failed = 0;
const check = (ok, what) => { console.log((ok ? 'ok   ' : 'FAIL ') + what); if (!ok) failed++; };

function machine(fight) {
  const m = fight ? startGame() : newGame(); installMawashi(m); installFlip(m);
  m.frames = []; m.lastFlip = m.video.page2;
  m.run = (s) => { const end = m.cpu.cycles + s * 1020484; while (m.cpu.cycles < end) { m.runTo(m.cpu.cycles + 1500);
    if (m.video.page2 !== m.lastFlip) { m.lastFlip = m.video.page2; let h = 0; const p = m.hgrPage(); for (let i = 0; i < p.length; i++) h = (h * 31 + p[i]) | 0; m.frames.push(h + ':' + m.cpu.cycles); } } };
  if (fight) { m.run(3); m.press('K'); m.run(0.5); m.press('SPACE'); m.run(3); }
  return m;
}
const play = (m, keys) => { for (const k of keys) { m.press(k); m.run(0.4); } };
const MORE = ['RIGHT', 'RIGHT', 'S', 'E', 'RIGHT', 'F', 'A', 'D', 'RIGHT', 'RIGHT', 'E', 'Q', 'X', 'F', 'RIGHT', 'W'];
const roundTrip = (m) => JSON.parse(JSON.stringify(saveState(m)));

function compare(a, what) {
  const snap = roundTrip(a), b = machine();
  b.sec(0.5); loadState(b, snap); b.lastFlip = a.lastFlip; b.frames = []; a.frames = [];
  play(a, MORE); play(b, MORE);
  const same = a.frames.length > 3 && a.frames.length === b.frames.length && a.frames.every((f, i) => f === b.frames[i]);
  check(same, `${what}: ${a.frames.length} frames identical after loading`);
  check(a.ram.every((v, i) => v === b.ram[i]) && a.cpu.cycles === b.cpu.cycles, `${what}: same memory and cycle count at the end`);
  check(a.flip.flips === b.flip.flips && a.mawashi.kicks === b.mawashi.kicks, `${what}: extensions agree (${b.flip.flips} flips, ${b.mawashi.kicks} kicks)`);
}

const a = machine(true);
play(a, ['RIGHT', 'RIGHT', 'RIGHT']);
compare(a, 'in the open');

const c = machine(true);
for (let i = 0; i < 20 && c.flip.frame < 2; i++) { c.press('F'); c.run(0.1); }
check(c.flip.frame >= 2, 'saving in the middle of a somersault');
compare(c, 'mid-somersault');

const t = machine(); t.sec(5);
compare(t, 'during the title sequence (disk spinning)');

let threw = false; try { loadState(machine(), { v: 99 }); } catch { threw = true; }
check(threw, 'a file that is not a save state is refused');

console.log(failed ? `${failed} failed` : 'all passed');
process.exit(failed ? 1 : 0);
