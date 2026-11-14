// Apple II hi-res page → canvas. 280x192, six colours, same rule the reference uses:
// a lit dot with a lit neighbour is white, a lone dot takes its colour from the column
// parity and the byte's palette bit, and a dark dot between two lit ones is filled in.
const ROW = new Uint16Array(192);
for (let y = 0; y < 192; y++) ROW[y] = ((y & 7) << 10) | (((y >> 3) & 7) << 7) | ((y >> 6) * 40);
export const PALETTE = [[0, 0, 0], [255, 255, 255], [255, 68, 253], [20, 245, 60], [20, 207, 253], [255, 106, 60]];

export class Display {
  constructor(canvas) {
    this.ctx = canvas.getContext('2d');
    this.img = this.ctx.createImageData(280, 192);
    this.px = new Uint32Array(this.img.data.buffer);
    this.rgba = PALETTE.map(([r, g, b]) => (255 << 24) | (b << 16) | (g << 8) | r);
    this.bits = new Uint8Array(282); this.hi = new Uint8Array(282);
  }
  draw(mem, base) {
    const { bits, hi, px, rgba } = this;
    for (let y = 0; y < 192; y++) {
      const row = base + ROW[y];
      for (let c = 0; c < 40; c++) {
        const b = mem[row + c];
        for (let i = 0; i < 7; i++) { bits[1 + c * 7 + i] = (b >> i) & 1; hi[1 + c * 7 + i] = b >> 7; }
      }
      for (let x = 0; x < 280; x++) {
        const on = bits[x + 1], l = bits[x], r = bits[x + 2];
        let col = 0;
        if (on) col = (l || r) ? 1 : (hi[x + 1] ? ((x & 1) ? 5 : 4) : ((x & 1) ? 3 : 2));
        else if (l && r) col = hi[x + 1] ? ((x & 1) ? 4 : 5) : ((x & 1) ? 2 : 3);
        px[y * 280 + x] = rgba[col];
      }
    }
    this.ctx.putImageData(this.img, 0, 0);
  }
}
