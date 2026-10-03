import * as P from '../js/physics.js';
import { ROCKETS } from '../js/rockets.js';
const gtoDv = (alt, inc) => { // from parking orbit to GTO apogee: just dv1 of hohmann
  const r1 = P.R_EARTH + alt; return P.hohmann(r1, P.GEO_RADIUS).dv1; };
const tliDv = alt => { const r1=P.R_EARTH+alt; const a=(r1+P.MOON_DIST)/2; return P.visViva(r1,a)-P.circularSpeed(r1); };
for (const id of Object.keys(ROCKETS)) {
  const R = ROCKETS[id]; const t0=Date.now();
  const inc = R.site.lat;
  const leo = P.maxPayload(R, 200000, 0, R.defaultInclination ?? inc);
  const gto = P.maxPayload(R, 200000, gtoDv(200000), inc);
  const tli = P.maxPayload(R, 185000, tliDv(185000), inc);
  console.log(id, 'LEO', (leo/1000).toFixed(1), 'GTO', (gto/1000).toFixed(1), 'TLI', (tli/1000).toFixed(1), 'published', JSON.stringify(R.published), (Date.now()-t0)+'ms');
}
console.log('gtoDv', gtoDv(200000).toFixed(0), 'tliDv', tliDv(185000).toFixed(0), 'GEO r', (P.GEO_RADIUS/1000).toFixed(1));
