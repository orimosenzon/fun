// Arbitrary-precision coordinates: fixed point on BigInt. A point of the complex plane is
// { x, y, bits } and its value is x / 2^bits, y / 2^bits. Only the centre of the view (or the
// place of the bike) is kept this way; everything else is a small double offset from it.

// how many fractional bits a view of this scale needs
export function bitsFor(scale) {
  const lg = Math.max(0, -Math.log2(Math.max(scale, 1e-310)));
  return 96 + Math.ceil(lg / 32) * 32;
}
export function shiftTo(v, from, to) {
  if (to === from) return v;
  return to > from ? v << BigInt(to - from) : v >> BigInt(from - to);
}
// double -> fixed
export function dToFix(x, bits) {
  if (x === 0 || !isFinite(x)) return 0n;
  const neg = x < 0; if (neg) x = -x;
  const e = Math.floor(Math.log2(x));
  const mi = BigInt(Math.round(x / 2 ** e * 2 ** 52));
  const sh = bits + e - 52;
  const r = sh >= 0 ? mi << BigInt(sh) : mi >> BigInt(-sh);
  return neg ? -r : r;
}
// fixed -> double (also for values far smaller than 2^-53)
export function fixToD(v, bits) {
  if (v === 0n) return 0;
  const neg = v < 0n; if (neg) v = -v;
  const bl = v.toString(2).length;
  const sh = bl - 53;
  const top = sh > 0 ? v >> BigInt(sh) : v << BigInt(-sh);
  const d = Number(top) * 2 ** (sh - bits);
  return neg ? -d : d;
}
// decimal string -> fixed
export function decToFix(s, bits) {
  s = String(s).trim();
  const neg = s.startsWith('-');
  if (neg || s.startsWith('+')) s = s.slice(1);
  if (/e/i.test(s)) return dToFix((neg ? -1 : 1) * Number(s), bits);
  const parts = s.split('.');
  const ip = parts[0] || '0', fp = parts[1] || '';
  const n = BigInt(ip + fp);
  const r = (n << BigInt(bits)) / 10n ** BigInt(fp.length);
  return neg ? -r : r;
}
export function fixToDec(v, bits, digits) {
  const neg = v < 0n; if (neg) v = -v;
  const scaled = (v * 10n ** BigInt(digits) + (1n << BigInt(bits - 1))) >> BigInt(bits);      // rounded, so -0.6 does not read -0.5999
  const s = scaled.toString().padStart(digits + 1, '0');
  return (neg && scaled !== 0n ? '-' : '') + s.slice(0, -digits) + '.' + s.slice(-digits);
}

// ----- points -----
export const bp = (xStr, yStr, bits) => ({ x: decToFix(xStr, bits), y: decToFix(yStr, bits), bits });
export function bpAlign(p, bits) {
  if (p.bits === bits) return p;
  return { x: shiftTo(p.x, p.bits, bits), y: shiftTo(p.y, p.bits, bits), bits };
}
export const bpAddD = (p, dx, dy) => ({ x: p.x + dToFix(dx, p.bits), y: p.y + dToFix(dy, p.bits), bits: p.bits });
// a - b as doubles
export function bpSubD(a, b) {
  const bits = Math.max(a.bits, b.bits);
  const A = bpAlign(a, bits), B = bpAlign(b, bits);
  return [fixToD(A.x - B.x, bits), fixToD(A.y - B.y, bits)];
}
export const bpToD = (p) => [fixToD(p.x, p.bits), fixToD(p.y, p.bits)];
