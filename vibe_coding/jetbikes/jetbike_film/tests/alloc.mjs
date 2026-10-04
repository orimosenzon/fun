import { FlightComputer } from '../js/control.js';
import { massProps, FUEL0 } from '../js/vehicle.js';
const fc = new FlightComputer();
const mp = massProps(FUEL0);
console.log('mass', mp.m.toFixed(1), 'cg', mp.cg.map(v=>v.toFixed(3)), 'I', mp.I.map(v=>v.toFixed(1)).join(' '));
for (const tau of [[0,0,0],[0,0,100],[100,0,0],[0,100,0]]) {
  const f = fc.allocate(3200, [0,0], tau, mp.cg, 1180);
  console.log('tau', tau, f.map(v=>v.map(x=>x.toFixed(0)).join('/')).join('  '));
}
