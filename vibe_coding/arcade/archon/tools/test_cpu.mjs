// node tools/test_cpu.mjs <6502_functional_test.bin>
// Klaus Dormann's functional test: success is an endless loop at $3469.
import fs from 'node:fs';
import { CPU6510 } from '../src/vm/cpu6510.js';
const mem = new Uint8Array(65536);
mem.set(fs.readFileSync(process.argv[2]));
const cpu = new CPU6510((a) => mem[a], (a, v) => { mem[a] = v; });
cpu.pc = 0x400;
let last = -1, n = 0;
for (; n < 200e6; n++) {
  if (cpu.pc === last) break;
  last = cpu.pc;
  cpu.step();
}
const ok = cpu.pc === 0x3469;
console.log(ok ? 'PASS' : 'FAIL', 'stopped at $' + cpu.pc.toString(16), 'after', n, 'instructions');
process.exit(ok ? 0 : 1);
