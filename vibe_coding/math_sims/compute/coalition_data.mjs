// מייצא את הניתוח של תשעת התרחישים של "מי יקים את הממשלה?" ל-data/coalition26.json.
// הפונקציות נלקחות מ-app.js עצמו (החלק שבין "מפלגות" ל"המקום בכנסת"), כך שהסרטון מציג בדיוק את מה שהאפליקציה מציגה.
// הרצה: node compute/coalition_data.mjs
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const APP = path.resolve(here, '../../coalition26');
const OUT = path.resolve(here, '../data/coalition26.json');

const ctx = { window: {} };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(APP, 'data.js'), 'utf8'), ctx);

const app = fs.readFileSync(path.join(APP, 'app.js'), 'utf8');
const a = app.indexOf('// ---------- מפלגות');
const b = app.indexOf('// ---------- המקום בכנסת');
if (a < 0 || b < 0) throw new Error('app.js changed: section markers not found');
const pure = app.slice(a, b);

vm.runInContext(`
const D = window.DATA;
const ROLE_W = { in: 1, support: 0.5, abstain: 0.2 };
const PM = { N: { party: 'L', name: 'בנימין נתניהו' }, E: { party: 'Y', name: 'גדי איזנקוט' } };
${pure}
const PLEDGE_PARTY = id => id === 'CH' ? 'גוש השינוי' : party(id).name;
window.result = D.scenarios.map(sc => {
  const st = { mode: sc.mode, roles: sc.roles };
  const t = tally(st);
  const out = outcomes(st);
  const broken = brokenPledges(st, out);
  const stab = stability(st, out, broken);
  return {
    id: sc.id, mode: sc.mode, title: sc.title, tag: sc.tag, lede: sc.lede, needs: sc.needs,
    roles: sc.roles, order: seatOrderPlain(st), tally: t, pivots: pivots(st),
    stability: { score: stab.score, label: stab.label, life: stab.life, cls: stab.cls, factors: stab.factors },
    broken: broken.map(({ p, sev }) => ({ party: p.party, partyName: PLEDGE_PARTY(p.party), text: p.text, sev })),
    outcomes: D.issues.map(iss => ({ id: iss.id, name: iss.name, minus: iss.minus, plus: iss.plus,
      value: out[iss.id], text: iss.outcomes[bucket(out[iss.id])],
      parties: Object.keys(sc.roles).filter(id => iss.pos[id] && iss.pos[id][1] > 0)
        .map(id => ({ id, pos: iss.pos[id][0], role: sc.roles[id] })) })),
    concessions: concessions(st, out).map(c => ({ id: c.id, role: c.role,
      wins: c.wins.map(w => w.want), comp: c.comp.map(w => w.want), gives: c.gives.map(w => w.want) })),
    tensions: tensions(st).filter(x => x.t >= 8).slice(0, 3).map(x => ({ a: x.a, b: x.b, issues: x.issues.slice(0, 3) }))
  };
});
// סדר המושבים בלי ה-DOM: ראש הממשלה, בקואליציה, תומכות, נמנעות, אופוזיציה
function seatOrderPlain(st) {
  const pm = PM[st.mode].party;
  const ins = active(st).filter(id => st.roles[id] === 'in' && id !== pm).sort((a, b) => seats(b) - seats(a));
  const sup = active(st).filter(id => st.roles[id] === 'support').sort((a, b) => seats(b) - seats(a));
  const abs = active(st).filter(id => st.roles[id] === 'abstain');
  const opp = idsFor(st.mode).filter(id => !(id in st.roles)).sort((a, b) => seats(a) - seats(b));
  return [[pm, 'in'], ...ins.map(id => [id, 'in']), ...sup.map(id => [id, 'support']),
    ...abs.map(id => [id, 'abstain']), ...opp.map(id => [id, 'opp'])];
}
`, ctx);

const D = ctx.window.DATA;
const parties = Object.fromEntries(D.parties.map(p => [p.id, { name: p.name, short: p.short, leader: p.leader, seats: p.seats, color: p.color[1] }]));
parties.Lx = { ...parties.L, name: D.likudAfter.name, short: D.likudAfter.short, leader: D.likudAfter.leader };
const json = { updated: D.updated, parties, blocs: D.blocs, scenarios: ctx.window.result };
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify(json, null, 1));
for (const s of json.scenarios) {
  console.log(`${s.title.padEnd(28)} in=${s.tally.inS} for=${s.tally.forS} against=${s.tally.against} pass=${s.tally.pass} stab=${s.stability.score} ${s.stability.label} broken=${s.broken.length}`);
}
