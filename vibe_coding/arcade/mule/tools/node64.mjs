// Node wrapper for M.U.L.E.: the C64 from ../archon/src/vm with the ROMs and the game disk from data/.
import fs from 'node:fs';
import { C64 } from '../../archon/src/vm/c64.js';
import { D64 } from '../../archon/src/vm/d64.js';
export { KEYS, JOY } from '../../archon/src/vm/c64.js';
export { writePNG, shot } from '../../archon/tools/node64.mjs';

const here = new URL('..', import.meta.url).pathname;
const bin = (f) => new Uint8Array(fs.readFileSync(f.startsWith('/') ? f : here + f));

export function makeMachine({ model = 'ntsc', disk = 'data/mule.d64' } = {}) {
  const m = new C64({ kernal: bin('data/kernal.bin'), basic: bin('data/basic.bin'), chargen: bin('data/chargen.bin'), model });
  if (disk) m.disk = new D64(bin(disk));
  return m;
}

// Boot and start the first program on the disk. The boot program loads the rest of the game
// from "the current device", so the machine has to believe it was loaded from drive 8.
export function startGame(opts) {
  const m = makeMachine(opts);
  if (!m.bootToReady()) throw new Error('BASIC did not come up');
  m.ram[0xba] = 8;
  m.runProgram(m.disk.load(Uint8Array.of(0x2a)));
  return m;
}
