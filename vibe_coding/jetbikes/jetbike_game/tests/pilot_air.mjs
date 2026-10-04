import { JetBike, makeWind } from '../js/physics.js';
import { PilotAssist } from '../js/pilot.js';
import { qrot, len } from '../js/vmath.js';
import { surfaceAt } from '../js/terrain.js';
const bike = new JetBike([0, 0, 0], 0); bike.p[1] = 700;; const fc = new PilotAssist(); const wind = makeWind();
bike.lift.forEach(e => { e.phase = 'run'; e.N = 0.34; }); bike.main.phase = 'run'; bike.main.N = 0.34;
const dt = 0.001;
const script = [ // [t_end, input]
  [10, {}], [22, { throttle: 1 }], [30, { throttle: 1, stickX: 1 }], [34, { throttle: 1 }],
  [38, { throttle: 1, stickY: 0.6 }], [44, { throttle: 1, stickY: -0.7 }], [48, { throttle: 0.3 }],
  [58, { throttle: 0, stickY: -1 }], [64, { throttle: 0 }], [70, { rudder: 1 }], [74, { stickY: 0.6 }], [80, {}],
  [86, { climb: -0.6 }],
];
let cmd = fc.cmd, k = 0, last = -1;
const inpAt = (t) => { for (const [te, i] of script) if (t < te) return { stickX: 0, stickY: 0, rudder: 0, climb: 0, throttle: 0, boost: false, assist: true, ...i }; return null; };
while (true) {
  const t = bike.t, inp = inpAt(t); if (!inp) break;
  if (k % 5 === 0) cmd = fc.update(dt * 5, bike, inp);
  bike.step(dt, cmd, wind); k++;
  if (Math.floor(t * 2) !== last) {
    last = Math.floor(t * 2); if (last % 2) continue;
    const f = qrot(bike.q, [1, 0, 0]), r = qrot(bike.q, [0, 0, 1]);
    const yaw = Math.atan2(-f[2], f[0]) * 57.3;
    console.log(`t=${t.toFixed(0).padStart(3)} ${JSON.stringify(inp).replace(/"assist":true,|"boost":false,|"(stickX|stickY|rudder|climb|throttle)":0,?/g, '').padEnd(30)} agl=${(bike.p[1] - 0.86 - surfaceAt(bike.p[0], bike.p[2])).toFixed(1).padStart(6)} vy=${bike.v[1].toFixed(1).padStart(5)} spd=${(len(bike.v) * 3.6).toFixed(0).padStart(4)} pitch=${(Math.asin(f[1]) * 57.3).toFixed(0).padStart(4)} bank=${(-Math.asin(r[1]) * 57.3).toFixed(0).padStart(4)} yaw=${yaw.toFixed(0).padStart(5)} T=${bike.lift.map(e => e.T.toFixed(0)).join('/')} m=${bike.main.T.toFixed(0)} Nc=${bike.contactN.toFixed(0)}`);
  }
}
