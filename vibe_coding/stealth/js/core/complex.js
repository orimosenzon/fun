// חשבון מרוכב מינימלי: מספר = [re, im]
export const cx = (re, im = 0) => [re, im];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
export const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
export const div = (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
export const abs = (a) => Math.hypot(a[0], a[1]);
export const scale = (a, s) => [a[0] * s, a[1] * s];
export function sqrt(a) { // ענף ראשי
  const r = abs(a), re = Math.sqrt(Math.max(0, (r + a[0]) / 2)), im = Math.sqrt(Math.max(0, (r - a[0]) / 2));
  return [re, a[1] < 0 ? -im : im];
}
export const exp = (a) => { const e = Math.exp(a[0]); return [e * Math.cos(a[1]), e * Math.sin(a[1])]; };
// tan(z) = -j·(e^{2jz}-1)/(e^{2jz}+1)
export function tan(z) {
  const e = exp([-2 * z[1], 2 * z[0]]);
  const q = div(sub(e, [1, 0]), add(e, [1, 0]));
  return [q[1], -q[0]];
}
