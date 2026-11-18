// node tools/boot_test.mjs: power on, print the BASIC screen.
import { makeMachine, shot } from './node64.mjs';
const m = makeMachine({ model: process.argv[2] || 'ntsc' });
const t = Date.now();
const ok = m.bootToReady();
console.log(ok ? 'READY' : 'NOT READY', 'after', m.vic.frames, 'frames,', Date.now() - t, 'ms, pc $' + m.cpu.pc.toString(16));
console.log(m.screenText().slice(0, 8).join('\n'));
shot(m, new URL('../reference/boot.png', import.meta.url).pathname);
