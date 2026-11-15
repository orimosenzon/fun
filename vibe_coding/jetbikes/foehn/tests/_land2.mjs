// Experiment: let go of the stick before the touch-down; taxi low; take off again. node tests/_land2.mjs
import { loadWorld } from './util.mjs';
import { Game } from '../js/sim/game.js';
import { buildScenery } from '../js/sim/scenery.js';
import { len } from '../js/sim/vmath.js';
loadWorld(); buildScenery();
const g = new Game(); g.load('first');
while (g.state !== 'fly') { g.update(1 / 60, {}); g.events.length = 0; }
g.cp = g.cps.findIndex((c) => c.type === 'land'); const c = g.cps[g.cp];
g.respawnPose = { p: [c.p[0] + 8, c.p[1] + 12, c.p[2]], yaw: Math.PI, speed: 0.01, pitch: 0 }; g.respawn(); g.vt = 0;
while (g.state !== 'fly') { g.update(1 / 60, {}); g.events.length = 0; }
const b = g.bike, K = { throttle: 0, turn: 0, strafe: 0, pitch: 0 };
const phase = (name, secs, f) => { let ev = ''; for (let i = 0; i < secs * 60; i++) { g.update(1 / 60, f(i / 60)); for (const e of g.events) if (['crash', 'land'].includes(e.type)) ev += ' ' + e.type + (e.reason || ''); g.events.length = 0; } console.log(`${name.padEnd(44)} agl=${b.agl.toFixed(2)} V=${len(b.v).toFixed(2)} vt=${g.vt.toFixed(1)} off=${Math.hypot(c.p[0] - b.p[0], c.p[2] - b.p[2]).toFixed(1)} idle=${g.sim.fcs.idle} low=${g.sim.fcs.low}${ev}`); };
phase('hover 2 s', 2, () => K);
let let_go = false;
phase('descend until 1 m, then let go', 7, (t) => { if (b.agl < 1.0) let_go = true; if (process.argv[2] && Math.round(t * 60) % 15 === 0) console.log(`   t=${t.toFixed(2)} in=${let_go ? 0 : -1} agl=${b.agl.toFixed(2)} vy=${b.v[1].toFixed(2)} vzDes=${g.sim.fcs.tele.vzDes.toFixed(2)} floor=${g.sim.fcs.tele.floor} low=${g.sim.fcs.low} N=${b.contactN.toFixed(0)}`); return { ...K, pitch: let_go ? 0 : -1 }; });
phase('taxi forward: W for 0.15 s, coast 2 s', 2.15, (t) => ({ ...K, throttle: t < 0.15 ? 1 : 0 }));
phase('brake 2 s', 2, () => ({ ...K, brake: true }));
phase('hold descend 1.5 s', 1.5, () => ({ ...K, pitch: -1 }));
phase('let go 1.5 s', 1.5, () => K);
phase('sit 2 s', 2, () => K);
phase('climb 2 s', 2, () => ({ ...K, pitch: 1 }));
phase('neutral 3 s (the floor is back)', 3, () => K);
