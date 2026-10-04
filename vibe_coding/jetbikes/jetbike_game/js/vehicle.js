// MJ-5 jet motorcycle: geometry, masses and engine data.
// Design frame ("D"): x forward, y up, z to the rider's right; origin on the ground plane
// under the middle of the bike (landing-pad soles at y = 0). The physics works about the
// centre of gravity, which moves slightly as fuel burns.
//
// Propulsion (numbers are in the range of real small turbojets such as the PBS TJ100 /
// JetCat P1000 class):
//   * 4 lift turbojets, vertical, exhausting DOWN through nozzles with two-axis vanes.
//     1300 N sea-level max thrust each, ~20 kg each. Pairs counter-rotate so their
//     gyroscopic moments cancel.
//   * 1 cruise turbojet on the centreline, exhausting BACKWARD through a normal tail pipe.
//     1900 N sea-level max thrust, 29 kg, tail pipe bent down so the thrust line passes the CG.

export const G = 9.80665;

export const LIFT_TMAX = 1300;   // N at sea level
export const MAIN_TMAX = 1900;   // N at sea level (bigger cruise engine than the film's: its thrust is canted)
export const VANE_MAX = 15 * Math.PI / 180;
export const TSFC = 4.3e-5;      // kg fuel per N per s  (0.155 kg/(N h), typical small turbojet)
export const EXHAUST_V = 560;    // m/s, used for intake momentum ("ram") drag
export const IDLE_N = 0.34;      // idle spool speed as a fraction of max rpm
export const THRUST_EXP = 3.2;   // thrust ~ N^3.2 (small single-spool turbojet)
export const ROTOR_I = 1.4e-3;   // kg m^2, lift engine rotor
export const MAIN_ROTOR_I = 2.2e-3;
export const OMEGA_MAX = 10500;  // rad/s (~100k rpm)
export const MAIN_OMEGA_MAX = 8900;
export const FUEL0 = 62;         // kg at start

// Lift engines: centre, nozzle exit, intake, rotor spin sign (about +y)
export const LIFT = [
  { name: 'FL', c: [0.80, 0.62, -0.41], spin: +1 },
  { name: 'FR', c: [0.80, 0.62, 0.41], spin: -1 },
  { name: 'RL', c: [-0.82, 0.62, -0.41], spin: -1 },
  { name: 'RR', c: [-0.82, 0.62, 0.41], spin: +1 },
].map((e) => ({ ...e, exit: [e.c[0], 0.30, e.c[2]], intake: [e.c[0], 0.95, e.c[2]] }));

// The cruise engine sits low under the seat, so a straight tail pipe would push its thrust
// line 35 cm below the centre of gravity (a 500 N m nose-up moment the lift jets would have
// to fight). Instead the tail pipe is bent 31.5 deg downward: the thrust line runs through
// the CG (solved numerically), and half of the cruise thrust helps carry the weight.
export const MAIN_CANT = 31.47 * Math.PI / 180;
export const MAIN_BEND = [-0.565, 0.52, 0];   // where the tail pipe bends
export const MAIN = { c: [-0.20, 0.52, 0], exit: [-1.0341, 0.2329, 0], intake: [0.45, 0.52, 0] };
export const MAIN_DIR = [Math.cos(MAIN_CANT), Math.sin(MAIN_CANT), 0];

export const PADS = [
  [0.52, 0, -0.48], [0.52, 0, 0.48], [-0.56, 0, -0.48], [-0.56, 0, 0.48],
];

// mass items: [mass kg, centre D, box size (x,y,z)]
const FIXED = [
  [48, [0.05, 0.78, 0], [1.9, 0.30, 0.34]],      // spine frame, fairings, wiring
  [14, [-0.40, 0.90, 0], [0.7, 0.12, 0.32]],     // seat, battery, flight computer
  [7, [0.38, 0.98, 0], [0.62, 0.30, 0.46]],      // tank shell
  [27, MAIN.c, [0.95, 0.30, 0.30]],              // cruise engine
  ...LIFT.map((e) => [20, e.c, [0.24, 0.62, 0.24]]),
  [9, [0, 0.25, 0], [1.1, 0.5, 0.96]],           // legs and pads
  // rider (82 kg incl. suit and helmet), seated, hands on the bars
  [37, [-0.30, 1.38, 0], [0.30, 0.60, 0.40]],    // torso
  [5.5, [-0.12, 1.84, 0], [0.26, 0.28, 0.24]],   // head + helmet
  [27, [0.02, 1.02, 0], [0.70, 0.30, 0.46]],     // legs
  [8.5, [0.12, 1.36, 0], [0.50, 0.14, 0.60]],    // arms
  [4, [-0.35, 1.05, 0], [0.30, 0.15, 0.36]],     // hips/boots remainder
];
const TANK = { c: [0.38, 0.98, 0], s: [0.60, 0.26, 0.42] };

export function massProps(fuel) {
  const items = [...FIXED, [fuel, TANK.c, TANK.s]];
  let m = 0;
  const cg = [0, 0, 0];
  for (const [mi, c] of items) { m += mi; cg[0] += mi * c[0]; cg[1] += mi * c[1]; cg[2] += mi * c[2]; }
  cg[0] /= m; cg[1] /= m; cg[2] /= m;
  const I = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (const [mi, c, s] of items) {
    const [a, b, d] = s;
    const x = c[0] - cg[0], y = c[1] - cg[1], z = c[2] - cg[2];
    I[0] += mi * (b * b + d * d) / 12 + mi * (y * y + z * z);
    I[4] += mi * (a * a + d * d) / 12 + mi * (x * x + z * z);
    I[8] += mi * (a * a + b * b) / 12 + mi * (x * x + y * y);
    I[1] -= mi * x * y; I[2] -= mi * x * z; I[5] -= mi * y * z;
  }
  I[3] = I[1]; I[6] = I[2]; I[7] = I[5];
  return { m, cg, I };
}

// thrust from spool speed (fraction of max rpm) and density ratio
export const thrustOfN = (N, tmax, sigma) => tmax * sigma * Math.pow(Math.max(N, 0), THRUST_EXP);
export const nOfThrust = (T, tmax, sigma) => Math.pow(Math.max(T, 0) / (tmax * sigma), 1 / THRUST_EXP);

// Aerodynamics of bike + rider (body frame): drag areas per axis and centres of pressure
export const AERO = {
  cda: [0.62, 1.45, 1.15],               // m^2 for flow along x (frontal), y (plan), z (side)
  cp: [[0, 0.30, 0], [-0.10, 0, 0], [-0.28, 0.12, 0]], // CP offsets from CG for each drag component
  rotDamp: [9, 14, 12],                  // N m s, aerodynamic rate damping
};
