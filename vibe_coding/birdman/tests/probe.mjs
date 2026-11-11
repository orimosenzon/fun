// torque breakdown at a frozen state: node tests/probe.mjs
import { newAcc, clearAcc, wingAero, bodyAero } from '../js/sim/aero.js';
const V = 13, al = 0.07;
function at(beta, foldR, foldL, twist, rud, omy = 0, omx = 0) {
  const vb = [V * Math.cos(al) * Math.cos(beta), -V * Math.sin(al), V * Math.sin(beta)];
  const om = [omx, omy, 0], z = [0, 0, 0], a = newAcc();
  const R = { phi: 0.1, phiDot: 0, fold: foldR, sweep: 0.1, twist: -twist, eta: 1 }, L = { ...R, fold: foldL, twist };
  wingAero(a, 1, R, vb, om, z, 0.6); const r = { ...a }; clearAcc(a);
  wingAero(a, -1, L, vb, om, z, 0.6); const l = { ...a }; clearAcc(a);
  bodyAero(a, { inc: 0, rud }, vb, om, 0.6, 0);
  const f = (o) => `F=${[o.fx, o.fy, o.fz].map((x) => x.toFixed(0))} T=${[o.tx, o.ty, o.tz].map((x) => x.toFixed(0))}`;
  console.log(`beta=${beta} foldR=${foldR} tw=${twist} rud=${rud} omy=${omy} omx=${omx}\n  R ${f(r)}\n  L ${f(l)}\n  body ${f(a)}\n  total T=${[r.tx + l.tx + a.tx, r.ty + l.ty + a.ty, r.tz + l.tz + a.tz].map((x) => x.toFixed(0))}`);
}
at(0, 1, 1, 0, 0); at(0.2, 1, 1, 0, 0); at(0.2, 1, 1, 0, 0.4); at(0, 0.58, 1, 0.08, 0); at(0, 1, 1, 0.1, 0); at(0, 1, 1, 0, 0, 0.3); at(0, 1, 1, 0, 0, 0, 0.4);
