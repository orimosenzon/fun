// ai.js: יריב ממוחשב. שומר מא-אי (מרחק), תוקף כשהטווח מתאים, חוסם לפי גובה ההתקפה ומחזיר נגד.

import { REACH } from './techniques.js';

const ATTACKS = [
  ['kizami', 'up', 'kizami_j', 3], ['kizami', 'down', 'kizami_c', 2],
  ['gyaku', 'mid', 'gyaku_c', 4], ['gyaku', 'up', 'gyaku_j', 2],
  ['oi', 'mid', 'oi_c', 3], ['oi', 'up', 'oi_j', 3],
  ['mae', 'mid', 'mae_c', 3], ['mae', 'up', 'mae_j', 1],
  ['mawashi', 'mid', 'mawashi_c', 2], ['mawashi', 'up', 'mawashi_j', 2],
  ['yoko', 'mid', 'yoko', 2], ['ushiro', 'mid', 'ushiro', 1],
];

const LEVELS = [
  { react: 0.36, block: 0.30, aggr: 0.45, counter: 0.3 },
  { react: 0.25, block: 0.55, aggr: 0.8, counter: 0.6 },
  { react: 0.17, block: 0.82, aggr: 1.2, counter: 0.9 },
];

export function makeAI(level = 1, rnd = Math.random) {
  const P = LEVELS[level];
  let think = 0.4, seen = -1, pending = null, pendT = 0;
  let moveDir = 0, moveT = 0, countered = -9, holdLevel = 'mid';

  return function ai(match, me, opp, dt) {
    const inp = { move: 0, level: 'mid', actions: [], hold: false };
    if (match.phase !== 'fight') return inp;
    const D = Math.abs(opp.x - me.x);
    const gap = D - 0.2;      // המרחק עד פני הגוף של היריב

    // תגובה להתקפה נכנסת
    if (opp.tech && opp.tech.kind === 'attack' && opp.techSeq !== seen) {
      seen = opp.techSeq;
      const r = REACH[opp.tech.id].x;
      if (gap < r + 0.15 && rnd() < P.block) {
        const cls = opp.tech.hit.cls;
        const lv = cls === 'J' ? 'up' : cls === 'G' ? 'down' : 'mid';
        if (rnd() < 0.18 && gap > r - 0.25) pending = { move: -1 };
        else pending = { a: cls === 'C' && rnd() < 0.35 ? 'shuto' : 'block', level: lv };
        pendT = Math.min(P.react * (0.7 + rnd() * 0.6), Math.max(0.02, opp.tech.hit.from - 0.1));
      }
    }
    if (pending) {
      pendT -= dt;
      if (pendT <= 0) {
        if (pending.move) { moveDir = -1; moveT = 0.4; }
        else { inp.actions.push(pending.a); inp.level = pending.level; holdLevel = pending.level; }
        pending = null;
      }
    }
    // מחזיקים את החסימה עד שההתקפה חולפת
    if (me.tech && me.tech.kind === 'block') {
      inp.level = holdLevel;
      inp.hold = !!(opp.tech && opp.tech.kind === 'attack' && !opp.hitDone && opp.t < opp.tech.hit.to);
      // התקפת נגד אחרי חסימה מוצלחת
      if (me.blockedAt > countered && match.time - me.blockedAt < 0.3) {
        countered = match.time;
        if (rnd() < P.counter) {
          const c = gap < REACH.gyaku_c.x ? ['gyaku', 'mid'] : gap < REACH.mae_c.x ? ['mae', 'mid'] : ['oi', 'mid'];
          inp.actions.push(c[0]); inp.level = c[1]; inp.hold = false;
        }
      }
      return inp;
    }

    if (moveT > 0) { moveT -= dt; inp.move = moveDir; }

    if (!me.busy && !pending) {
      think -= dt;
      if (think <= 0) {
        think = 0.10 + rnd() * 0.22;
        const oppOpen = opp.tech && (opp.tech.kind === 'react' || (opp.tech.kind === 'attack' && opp.hitDone));
        const opts = ATTACKS.filter(([, , id]) => REACH[id].x > gap + 0.04 && REACH[id].x < gap + 0.5);
        const p = (oppOpen ? 0.75 : 0.28) * P.aggr;
        if (opts.length && rnd() < p) {
          let tot = 0; for (const o of opts) tot += o[3];
          let r = rnd() * tot, pick = opts[0];
          for (const o of opts) { r -= o[3]; if (r <= 0) { pick = o; break; } }
          inp.actions.push(pick[0]); inp.level = pick[1];
          moveT = 0;
        } else if (gap > 1.45) { moveDir = 1; moveT = 0.2 + rnd() * 0.3; }
        else if (gap < 0.75 && rnd() < 0.5) { moveDir = -1; moveT = 0.2 + rnd() * 0.2; }
        else if (rnd() < 0.3) { moveDir = rnd() < 0.55 * P.aggr ? 1 : -1; moveT = 0.15 + rnd() * 0.2; }
      }
    }
    return inp;
  };
}
