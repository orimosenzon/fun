// node tests/field.mjs : the CPU side of the field against plain arithmetic.
import { Field, evalDirect, iterCount } from '../js/core/field.js';
import { bp, bpAddD, bpSubD, bpAlign, decToFix, fixToDec, fixToD, dToFix, bitsFor } from '../js/core/fixed.js';

let fail = 0;
const ok = (name, cond, info = '') => { console.log((cond ? 'ok   ' : 'FAIL ') + name + ' ' + info); if (!cond) fail++; };

// ---- fixed point ----
{
  const a = decToFix('-0.743643887037158704752191506114774', 256);
  ok('decimal round trip', fixToDec(a, 256, 30) === '-0.743643887037158704752191506115');
  ok('double round trip', Math.abs(fixToD(dToFix(-1.2345678901234e-27, 192), 192) / -1.2345678901234e-27 - 1) < 1e-12);
  const p = bp('0.25', '-0.5', 128), q = bpAddD(p, 3e-30, -7e-31);
  const [dx, dy] = bpSubD(q, p);
  ok('tiny offsets survive', Math.abs(dx / 3e-30 - 1) < 1e-6 && Math.abs(dy / -7e-31 - 1) < 1e-6, `${dx} ${dy}`);
}

// ---- perturbation against direct iteration, at scales where doubles are still exact enough ----
{
  const F = new Field(), o = {}, r = {};
  let worstSl = 0, worstD = 0, n = 0, inside = 0;
  const spots = [['-0.6', '0', 1.35], ['-0.74453986046', '0.12172377365', 3e-4], ['0.28693186889', '0.01428669390', 2e-6], ['-0.0886813257', '0.6545058352', 4e-6], ['-1.7864402555', '0', 4e-9], ['0.9', '0.7', 0.5]];
  for (const [x, y, s] of spots) {
    const it = iterCount(s, 1.5);
    const c = bp(x, y, bitsFor(s));
    F.build(c, s, it);
    const cx = Number(x), cy = Number(y);
    for (let k = 0; k < 400; k++) {
      const dx = (Math.sin(k * 12.9898) * 43758.5453 % 1) * s * 1.6, dy = (Math.sin(k * 78.233) * 12345.678 % 1) * s;
      F.eval(dx, dy, it, o); evalDirect(cx + dx, cy + dy, it, r);
      if ((o.sl < 0) !== (r.sl < 0)) { if (s > 1e-7) { worstSl = 99; } continue; }   // (at 4e-9 a double c is too coarse for the yardstick itself)
      if (o.sl < 0) { inside++; continue; }
      if (r.d < s * 1e-6) continue;       // closer to the set than a millionth of the view: doubles cannot tell, either way
      n++;
      const tol = s > 1e-7 ? 1 : 0;
      if (tol) { worstSl = Math.max(worstSl, Math.abs(o.sl - r.sl)); worstD = Math.max(worstD, Math.abs(o.d / r.d - 1)); }
    }
  }
  ok('perturbation = direct (smooth count)', worstSl < 1e-3, `worst ${worstSl.toExponential(2)} over ${n} points (${inside} inside)`);
  ok('perturbation = direct (distance)', worstD < 1e-3, `worst relative ${worstD.toExponential(2)}`);
}

// ---- a reference far from the points (rebasing must carry them) ----
{
  const F = new Field(), o = {}, r = {};
  F.build(bp('-0.5', '0', 128), 1, 600);          // inside the set: the orbit never escapes
  let w = 0, wd = 0;
  for (let k = 0; k < 300; k++) {
    const dx = -1.6 + 3.1 * (k % 17) / 17, dy = -1.2 + 2.4 * (k % 13) / 13;
    F.eval(dx, dy, 600, o); evalDirect(-0.5 + dx, dy, 600, r);
    if (o.sl < 0 || r.sl < 0) { if ((o.sl < 0) !== (r.sl < 0)) w = 99; continue; }
    w = Math.max(w, Math.abs(o.sl - r.sl)); wd = Math.max(wd, Math.abs(o.d / r.d - 1));
  }
  ok('reference inside the set, points all over', w < 1e-6 && wd < 1e-6, `${w.toExponential(2)} ${wd.toExponential(2)}`);
  F.build(bp('1.5', '1.5', 128), 1, 600);         // far outside: the orbit is two terms long
  w = 0;
  for (let k = 0; k < 300; k++) {
    const x = -1.9 + 2.4 * (k % 17) / 17, y = -1.1 + 2.2 * (k % 13) / 13;
    F.eval(x - 1.5, y - 1.5, 600, o); evalDirect(x, y, 600, r);
    if (o.sl < 0 || r.sl < 0) { if ((o.sl < 0) !== (r.sl < 0)) w = 99; continue; }
    w = Math.max(w, Math.abs(o.sl - r.sl));
  }
  ok('reference that escapes at once', w < 1e-6, w.toExponential(2));
}

// ---- deep: against exact BigInt iteration of the point itself ----
{
  const deep = (xs, ys, s, label, spread = 1 / 30) => {
    const it = 45000, bits = bitsFor(s) + 64;
    const c = bp(xs, ys, bits);
    const F = new Field(); F.build(c, s, it);
    let bad = 0, n = 0, esc = 0;
    const o = {};
    for (let k = 0; k < 24; k++) {
      const dx = Math.cos(k * 2.4) * s * (0.2 + k * spread), dy = Math.sin(k * 2.4) * s * (0.2 + k * spread);
      F.eval(dx, dy, it, o);
      // exact escape time of c + d
      const p = bpAddD(bpAlign(c, bits), dx, dy), B = BigInt(bits), bail = 4096n << B;
      let zx = 0n, zy = 0n, e = -1;
      for (let i = 1; i <= it; i++) { const t = ((zx * zx - zy * zy) >> B) + p.x; zy = ((zx * zy) >> (B - 1n)) + p.y; zx = t; if (((zx * zx + zy * zy) >> B) > bail) { e = i; break; } }
      n++;
      if (e < 0) { if (o.sl >= 0) bad++; continue; }
      if (o.sl >= 0 && o.d < s * 1e-3) continue;       // a hair from the set: the last digits decide
      esc++;
      // the smooth count lies within about one of the escape iteration
      if (o.sl < 0 || Math.abs(o.sl - SLCbase(e)) > 1.6) bad++;
      if (!(o.d > 0 && o.d < s * 1e6)) bad++;
    }
    ok(label, bad === 0 && esc > 0, `${esc}/${n} escape, ${bad} wrong, ${it} iterations, ${bits} bits`);
  };
  const SLCbase = (e) => e + 1;
  deep('-0.743643887037158704752191506114774', '0.131825904205311970493132056385139', 1.2e-13, 'depth 1e-13 = exact iteration');
  deep('-0.743643887037158704752191506114774', '0.131825904205311970493132056385139', 1e-26, 'depth 1e-26 = exact iteration', 4);
  deep('-0.743643887037158704752191506114774', '0.131825904205311970493132056385139', 3e-31, 'depth 3e-31 = exact iteration', 4);
}

// ---- the distance estimate is a distance: along a ray from the real axis tip it shrinks to 0 ----
{
  const F = new Field(), o = {};
  F.build(bp('-2', '0', 128), 1, 4000);
  let good = true, last = 1e9;
  for (const x of [0.5, 0.2, 0.1, 0.03, 0.01, 0.003, 0.001]) {      // c = -2 - x: the set ends at -2
    F.eval(-x, 0, 4000, o);
    if (!(o.d > 0.9 * x && o.d < 2.1 * x && o.d < last)) good = false;   // at the tip of a needle the estimate is twice the distance (the extreme case of Koebe's theorem)
    last = o.d;
  }
  ok('distance to the tip at -2', good, `d(-2.001) = ${last.toExponential(3)}`);
}

// ---- skipping: the same answers in a fraction of the steps ----
{
  const F = new Field(), a = {}, b = {};
  F.build(bp('-0.743643887037158704752191506114774', '0.131825904205311970493132056385139', 320), 1e-30, 45000);
  let w = 0, sa = 0, sb = 0, n = 0;
  for (const s of [1e-12, 1e-20, 1e-28]) for (let k = 0; k < 150; k++) {
    const dx = Math.cos(k * 1.7) * s * (k / 75), dy = Math.sin(k * 1.7) * s * (k / 75);
    F.useBla = true; F.eval(dx, dy, 45000, a); F.useBla = false; F.eval(dx, dy, 45000, b); F.useBla = true;
    sa += a.steps; sb += b.steps;
    if (a.sl < 0 || b.sl < 0 || b.d < s * 1e-2) continue;
    n++; w = Math.max(w, Math.abs(a.sl - b.sl));
  }
  ok('skipping keeps the smooth count', w < 0.02 && n > 100, `worst ${w.toExponential(2)} over ${n} points`);
  ok('skipping saves steps', sa * 8 < sb, `${Math.round(sb / sa)} times fewer`);
}

console.log(fail ? `${fail} FAILED` : 'ALL PASS');
process.exit(fail ? 1 : 0);
