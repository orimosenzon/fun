// node tools/dir.mjs <image.d64>: list the directory with load addresses.
import fs from 'node:fs';
import { D64 } from '../src/vm/d64.js';
const d = new D64(new Uint8Array(fs.readFileSync(process.argv[2])));
for (const e of d.directory()) {
  const b = d.readChain(e.track, e.sector);
  console.log(String(e.blocks).padStart(4), JSON.stringify(e.text).padEnd(20), e.type, '$' + (b[0] | (b[1] << 8)).toString(16).padStart(4, '0'), b.length - 2, 'bytes');
}
