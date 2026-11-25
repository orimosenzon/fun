/* The flying car over the map: the flight itself, with nothing drawn.
 *
 * The car is the Tesla Roadster of the roadster project (../../roadster),
 * and in life it does not fly: its cold-gas thrusters hold it in a hover for
 * a few seconds and the tanks are empty. Ori, 25/11/2026: "let us bend
 * reality a little. Assume that instead of jets of compressed air it has the
 * mechanism of the bike that runs on cold fusion." That bike is FÖHN's F-5,
 * and so the flight here is the F-5's, whole: its fusion core, its four lift
 * nozzles where the wheels are and its rear nozzle, and the flight computer
 * that turns "that way, this fast, this high" into valves. Nothing of it is
 * copied; CarFlight is F5 (f5_core.js) with a different hand on it.
 *
 * What is the car's own is how it is flown, and that is two things. A car
 * is two metres wide and four long, and thrown onto its side the way the
 * bike is thrown it looks like a crash that has not happened yet; so the
 * steepest bank is shallower and it rolls into it more slowly. The numbers
 * are FÖHN's own dials (fcs.js, FEEL), turned for the length of a step and
 * turned back, so the F-5 a moment later is still the F-5.
 *
 * And the pulses: Z and X are still a side-step and Shift a hop, as on the
 * F-5, because they are the same nozzles.
 *
 * No three.js in here, on purpose: tests/test_car.mjs flies this file in Node.
 */
import { F5, yawFor, V_MAX, FLAT } from './f5_core.js';
import { FEEL } from '../../jetbikes/foehn/js/sim/fcs.js';

export { yawFor, V_MAX, FLAT };

/** FÖHN's dials as the car has them. The bike banks to 73 degrees and gets
 *  there at 3.2 rad/s; the car to 52, at 2.2, which at 180 km/h is still a
 *  turn of two hundred metres across. */
export const CAR_FEEL = {
  bank: 0.9,
  rollRate: 2.2,
  rollAcc: 9,
  yawRate: 1.5,
  yawAcc: 4.5,
  strafe: 7
};

export class CarFlight extends F5 {
  step(dt) {
    const was = {};
    for (const k of Object.keys(CAR_FEEL)) { was[k] = FEEL[k]; FEEL[k] = CAR_FEEL[k]; }
    try {
      super.step(dt);
    } finally {
      Object.assign(FEEL, was);
    }
  }
}
