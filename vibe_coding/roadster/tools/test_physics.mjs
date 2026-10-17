// בדיקות פיזיקה ב-node: node tools/test_physics.mjs [שם]
import { World } from '../js/collide.js';
import { buildTrack, SPAWNS, LAYOUT, wallProfile, loopGuide, wallRideProfile } from '../js/track.js';
import { Car } from '../js/car.js';
import { Controller } from '../js/control.js';
import { V3, G0 } from '../js/math.js';
import { Nozzle, AirTank, PSI, BAR } from '../js/gas.js';
import { solveFan } from '../js/fan.js';

const world = new World();
const track = buildTrack(world);
const DT = 1 / 600;
const only = process.argv[2];
const T = (name, fn) => { if (!only || name.includes(only)) { console.log(`\n== ${name}`); const t0 = performance.now(); fn(); console.log(`   (${((performance.now() - t0) / 1000).toFixed(1)}s)`); } };

function sim(car, ctl, seconds, rawFn, onStep) {
  const n = Math.round(seconds / DT);
  for (let i = 0; i < n; i++) {
    const raw = { throttle: 0, brake: 0, steer: 0, boost: 0, lift: 0, pitch: 0, roll: 0, yaw: 0, ...rawFn(car, i * DT) };
    if (i % 10 === 0) ctl.update(car, raw, DT * 10, { altitude: car.altitude, guide: loopGuide });
    car.step(DT, ctl.cmd);
    if (onStep && onStep(car, i * DT) === false) break;
  }
}

T('wall profile', () => {
  const pr = wallProfile();
  const end = pr.at(pr.total);
  console.log(`   length ${pr.total.toFixed(1)} m, top height ${end.v.toFixed(1)} m, final angle ${(end.th * 180 / Math.PI).toFixed(0)}°, overhang h=${end.h.toFixed(1)}`);
  let maxH = 0; for (const p of pr.pts) maxH = Math.max(maxH, p.v);
  console.log(`   max height ${maxH.toFixed(1)} m`);
});

T('nozzles & tank', () => {
  for (const [F, name] of [[20000, 'main'], [12000, 'corner'], [4000, 'yaw']]) {
    const At = Nozzle.sizeFor(F, 5500 * PSI, 293.15);
    const nz = new Nozzle({ throatArea: At });
    const p = nz.perf(5500 * PSI, 293.15);
    console.log(`   ${name}: throat d=${(Math.sqrt(At * 4 / Math.PI) * 1000).toFixed(1)} mm, F=${(p.F / 1000).toFixed(1)} kN, mdot=${p.mdot.toFixed(1)} kg/s, Isp=${(p.F / p.mdot / G0).toFixed(0)} s, Me=${nz.Me.toFixed(2)}, pe=${(p.pe / BAR).toFixed(1)} bar (${p.regime})`);
  }
  const tank = new AirTank({ volume: 0.6, pFill: 5500 * PSI });
  console.log(`   tank 600 L @ 379 bar: ${tank.m.toFixed(0)} kg air, exergy ${(tank.exergy() / 3.6e6).toFixed(1)} kWh`);
  // ריקון בדחף מלא של 4 הראשיים
  const nz = new Nozzle({ throatArea: Nozzle.sizeFor(20000, 5500 * PSI, 293.15) });
  let t = 0, imp = 0; const rows = [];
  while (tank.m > 1 && t < 20) {
    const p = nz.perf(tank.p, tank.T);
    const F = 4 * p.F;
    if (F < 1000) break;
    imp += F * 0.001;
    tank.step(4 * p.mdot, 0.001); t += 0.001;
    if (Math.abs(t % 0.5) < 0.0005) rows.push(`t=${t.toFixed(1)} p=${(tank.p / BAR).toFixed(0)}bar T=${(tank.T - 273.15).toFixed(0)}C F=${(F / 1000).toFixed(0)}kN m=${tank.m.toFixed(0)}`);
  }
  console.log('   ' + rows.join('\n   '));
  console.log(`   blowdown to <1kN: ${t.toFixed(2)} s, impulse ${(imp / 1000).toFixed(0)} kN·s`);
});

T('fan', () => {
  for (const gap of [0.002, 0.003, 0.005, 0.01, 0.02, 0.05]) {
    const s = solveFan(1, 9.2 * gap);
    console.log(`   gap ${gap * 1000} mm: dp=${(s.dp / 1000).toFixed(1)} kPa, force=${(s.dp * 3.8 / G0).toFixed(0)} kgf, Q=${s.Q.toFixed(1)} m³/s, P=${(s.power / 1000).toFixed(0)} kW, n=${s.n.toFixed(2)}${s.powerLimited ? ' (power limited)' : ''}`);
  }
});

T('rest', () => {
  const car = new Car('rocket', world); const ctl = new Controller();
  car.reset(0, -10, 0);
  sim(car, ctl, 3, () => ({}));
  console.log(`   cg y=${car.pos.y.toFixed(3)} contacts=${car.contacts} v=${car.speed.toFixed(4)} comp=${car.wheels.map(w => w.comp.toFixed(3)).join(',')} g=${car.gBody.y.toFixed(2)} fan=${(car.fan.force / G0).toFixed(0)}kgf`);
});

for (const vk of ['performance', 'thrust', 'rocket']) T(`0-60 ${vk}`, () => {
  const car = new Car(vk, world); const ctl = new Controller(); ctl.fanMode = 'max';
  car.reset(LAYOUT.drag.x, LAYOUT.drag.z0, 0);
  sim(car, ctl, 1.5, () => ({ brake: 0 })); // מאווררים מסתובבים
  let t60 = null, t100 = null, tq = null, x0 = car.pos.z, maxG = 0, airUsed = car.tank.m;
  sim(car, ctl, 14, (c, t) => ({ throttle: 1, boost: 1 }), (c, t) => {
    const v = c.vel.z;
    maxG = Math.max(maxG, c.gBody.z);
    if (t60 === null && v >= 26.8224) { t60 = t; airUsed -= c.tank.m; }
    if (t100 === null && v >= 27.7778) t100 = t;
    if (tq === null && c.pos.z - x0 >= 402.336) { tq = t; console.log(`   1/4 mile ${t.toFixed(2)} s @ ${(v * 3.6).toFixed(0)} km/h`); }
  });
  console.log(`   0-60 mph ${t60?.toFixed(2)} s (claim <${car.P.claimed060}), 0-100 km/h ${t100?.toFixed(2)} s, peak ${maxG.toFixed(2)} g, air used to 60: ${airUsed.toFixed(0)} kg, v=${(car.speed * 3.6).toFixed(0)} km/h`);
});

T('top speed', () => {
  const car = new Car('rocket', world); const ctl = new Controller(); ctl.fanMode = 'off';
  car.reset(450, -500, 0);
  sim(car, ctl, 40, () => ({ throttle: 1 }));
  console.log(`   v=${(car.speed * 3.6).toFixed(0)} km/h, z=${car.pos.z.toFixed(0)}`);
});

T('skidpad', () => {
  for (const fm of ['off', 'max']) {
    const car = new Car('rocket', world); const ctl = new Controller(); ctl.fanMode = fm; ctl.aeroMode = 'high';
    car.reset(-200, -300, 0);
    // מאיצים ואז שומרים על מהירות קבועה עם היגוי מלא בהדרגה; מודדים את התאוצה הצידית המרבית שמחזיקה
    let best = 0;
    for (const vt of [12, 15, 18, 21, 24, 27, 30]) {
      car.reset(0, -200, 0); car.vel.set(0, 0, vt);
      for (const w of car.wheels) w.omega = vt / car.P.wheelRadius;
      let latMax = 0, spun = false;
      sim(car, ctl, 4, (c) => { const v = c.vel.dot(c.fwd); return { throttle: v < vt ? 0.6 : 0.15, steer: 0.45 }; }, (c, t) => {
        if (t > 2) {
          const yawRate = c.w.y, v = c.speed;
          const slide = Math.abs(Math.atan2(c.vel.dot(c.left), c.vel.dot(c.fwd)));
          if (slide > 0.25) spun = true;
          latMax = Math.max(latMax, Math.abs(yawRate * v) / G0);
        }
      });
      if (!spun) best = Math.max(best, latMax);
    }
    console.log(`   fan ${fm}: max steady lateral ≈ ${best.toFixed(2)} g`);
  }
});

T('wall', () => {
  for (const speed of [8, 11, 14, 17, 20]) {
    const car = new Car('rocket', world); const ctl = new Controller(); ctl.fanMode = 'max';
    const [x, z] = SPAWNS.wall;
    car.reset(x, LAYOUT.wall.z - 60, 0);
    let maxPitch = 0, hAt = 0, dfAt = 0, minContacts = 4, onWall = true, peakG = 0, landed = false, landG = 0, maxH = 0;
    let released = false, vTouch = 0;
    sim(car, ctl, 14, (c) => {
      const v = c.vel.dot(c.fwd);
      return { throttle: c.contacts > 0 && v < speed ? Math.min(1, (speed - v) * 0.5 + 0.3) : 0 };
    }, (c, t) => {
      const p = c.pitchDeg;
      maxH = Math.max(maxH, c.pos.y);
      if (c.contacts > 0 && c.pos.z > LAYOUT.wall.z) {
        if (p > maxPitch || (p < -100 && maxPitch > 90)) {
          const pp = p < -100 ? 360 + p : p;
          if (pp > maxPitch) { maxPitch = pp; hAt = c.pos.y; dfAt = c.fan.force / G0; }
        }
      }
      if (maxPitch > 100 && c.contacts === 0) released = true;
      if (released) { peakG = Math.max(peakG, c.gSmooth.len()); if (c.contacts === 0) vTouch = c.vel.y; }
      if (released && c.contacts >= 3 && c.pos.y < 2) { landed = true; landG = peakG; return false; }
    });
    car.basis();
    console.log(`   v=${speed} m/s: max pitch ${maxPitch.toFixed(0)}° at h=${hAt.toFixed(1)} m, downforce ${dfAt.toFixed(0)} kgf, maxH ${maxH.toFixed(1)}, released=${released}, landed=${landed} upright=${car.up.y.toFixed(2)} peakG=${landG.toFixed(1)} vTouch=${vTouch.toFixed(1)} air left ${car.tank.m.toFixed(0)} kg, crashImpact ${car.maxImpact.toFixed(1)}`);
  }
});

T('jump', () => {
  const car = new Car('rocket', world); const ctl = new Controller();
  car.reset(0, -100, 0);
  sim(car, ctl, 1, () => ({}));
  const m0 = car.tank.m; let maxH = 0;
  sim(car, ctl, 6, (c, t) => ({ lift: t < 0.7 ? 1 : 0 }), (c) => { maxH = Math.max(maxH, c.pos.y - c.P.cgHeight); });
  console.log(`   0.7 s lift: max height ${maxH.toFixed(1)} m, air used ${(m0 - car.tank.m).toFixed(0)} kg, up=${car.up.y.toFixed(2)}, contacts ${car.contacts}, impact ${car.maxImpact.toFixed(1)}`);
});

T('kickflip', () => {
  const car = new Car('rocket', world); const ctl = new Controller();
  car.reset(LAYOUT.lineup.x, LAYOUT.lineup.z - 60, 0);
  let started = false, maxH = 0, landed = false, hitCars = false;
  sim(car, ctl, 12, (c) => {
    const v = c.vel.dot(c.fwd);
    if (!started && c.pos.z > LAYOUT.lineup.z - 16) { started = true; ctl.startKickflip(1); }
    return { throttle: v < 16 ? 1 : 0.1 };
  }, (c) => {
    maxH = Math.max(maxH, c.pos.y);
    if (c.hullContact && c.pos.y > 0.8) hitCars = true;
    if (started && !ctl.kick && c.contacts >= 3 && c.pos.z > track.lineupEnd) { landed = true; return false; }
  });
  car.basis();
  console.log(`   max cg height ${maxH.toFixed(1)} m, landed past lineup=${landed}, z=${car.pos.z.toFixed(1)} (end ${track.lineupEnd.toFixed(1)}), up=${car.up.y.toFixed(2)}, hit=${hitCars}, impact=${car.maxImpact.toFixed(1)}, air left ${car.tank.m.toFixed(0)} kg`);
});

T('tunnel', () => {
  const car = new Car('rocket', world); const ctl = new Controller(); ctl.fanMode = 'max';
  const L = LAYOUT.tunnel;
  car.reset(L.x, L.z0 - 40, 0);
  let maxH = 0;
  sim(car, ctl, 9, (c, t) => {
    const v = c.vel.dot(c.fwd);
    return { throttle: v < 16 ? 1 : 0.3, steer: c.pos.z > L.z0 + 5 ? 0.35 : 0 };
  }, (c) => { if (c.pos.z > L.z0 && c.pos.z < L.z1) maxH = Math.max(maxH, c.pos.y); });
  car.basis();
  console.log(`   max cg height in tunnel ${maxH.toFixed(1)} m (diam ${2 * L.r}), up=${car.up.y.toFixed(2)}, contacts ${car.contacts}, z=${car.pos.z.toFixed(0)}`);
});

T('loop', () => {
  for (const v0 of [14, 18, 22]) {
    const car = new Car('rocket', world); const ctl = new Controller(); ctl.fanMode = 'max';
    const L = LAYOUT.loop;
    car.reset(L.x, L.z - 70, 0);
    let maxH = 0, minUp = 1, lost = 0;
    sim(car, ctl, 12, (c) => ({ throttle: c.vel.dot(c.fwd) < v0 ? 1 : 0.2 }), (c) => {
      maxH = Math.max(maxH, c.pos.y); minUp = Math.min(minUp, c.up.y);
      if (c.pos.y > 2 && c.contacts === 0) lost++;
    });
    car.basis();
    console.log(`   v=${v0}: max h ${maxH.toFixed(1)}, min up.y ${minUp.toFixed(2)}, airborne steps ${lost}, end x=${car.pos.x.toFixed(1)} z=${car.pos.z.toFixed(0)} up=${car.up.y.toFixed(2)}`);
  }
});

T('wallride', () => {
  const car = new Car('rocket', world); const ctl = new Controller(); ctl.fanMode = 'max';
  const L = LAYOUT.wallRide;
  car.reset(L.x + 8, -200, 0);
  let maxH = 0;
  sim(car, ctl, 14, (c, t) => {
    const v = c.vel.dot(c.fwd);
    // היגוי ימינה אל הקיר, ואז שמירה על גובה
    const steer = t < 3 ? 0 : t < 3.6 ? -0.5 : (c.pos.y < 7 ? -0.12 : 0.1);
    return { throttle: v < 20 ? 1 : 0.3, steer };
  }, (c) => { maxH = Math.max(maxH, c.pos.y); });
  car.basis();
  console.log(`   max h ${maxH.toFixed(1)}, final y ${car.pos.y.toFixed(1)}, up=(${car.up.x.toFixed(2)},${car.up.y.toFixed(2)}), contacts ${car.contacts}`);
});

T('wall hold', () => {
  // רכב על הקיר האנכי, נוסע אופקית. האם היניקה מחזיקה אותו?
  for (const [fm, v0] of [['max', 20], ['off', 20], ['max', 0]]) {
    const car = new Car('rocket', world); const ctl = new Controller(); ctl.fanMode = fm;
    const L = LAYOUT.wallRide, pr = wallRideProfile(L), q = pr.at(pr.total - 5);
    car.reset(0, 0, 0);
    car.pos.set(L.x - q.h + car.P.cgHeight, q.v, -100);
    car.q.setFromBasis(new V3(0, -1, 0), new V3(1, 0, 0), new V3(0, 0, 1));
    car.vel.set(0, 0, v0); for (const w of car.wheels) w.omega = v0 / car.P.wheelRadius;
    car.fan.n = fm === 'max' ? 1 : 0;
    const y0 = car.pos.y;
    sim(car, ctl, 3, (c) => ({ throttle: c.vel.z < v0 ? 0.5 : 0.1, steer: 0 }));
    console.log(`   fan ${fm} v0=${v0}: dropped ${(y0 - car.pos.y).toFixed(2)} m in 3 s, contacts ${car.contacts}, downforce ${(car.fan.force / G0).toFixed(0)} kgf, up.x=${car.up.x.toFixed(2)}`);
  }
});

T('hover', () => {
  const car = new Car('rocket', world); const ctl = new Controller();
  car.reset(LAYOUT.hover.x, LAYOUT.hover.z, 0);
  sim(car, ctl, 0.5, () => ({}));
  let tHover = 0, yMin = 99, yMax = 0, started = false;
  sim(car, ctl, 20, (c, t) => {
    if (t > 0.9 && !ctl.hoverHold && !started) { ctl.hoverHold = true; started = true; }
    return { lift: t < 0.9 ? 0.6 : 0 };
  }, (c, t) => {
    if (started && c.contacts === 0 && ctl.hoverHold) { tHover += DT; yMin = Math.min(yMin, c.pos.y); yMax = Math.max(yMax, c.pos.y); }
    if (started && c.contacts >= 3 && t > 2) return false;
  });
  console.log(`   hover hold ${tHover.toFixed(1)} s, y ${yMin.toFixed(1)}..${yMax.toFixed(1)} m, air left ${car.tank.m.toFixed(0)} kg (${(car.tank.p / 1e5).toFixed(0)} bar), landed impact ${car.maxImpact.toFixed(1)}`);
});
