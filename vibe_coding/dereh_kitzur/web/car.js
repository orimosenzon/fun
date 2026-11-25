/* The flying car - the flight mode's fifth vehicle (25/11/2026).
 *
 * This is the door to it and nothing more, as bike5.js is the F-5's. The car
 * is an ES module (car_rig.js) that imports the roadster project's model and
 * FÖHN's flight, so it is fetched on the first take-off and not on page
 * load. The handles are bike.js's, so explore.js can hold any of the three.
 */
'use strict';

const CarFly = (() => {
  let mod = null;
  let loading = null;

  /** Fetch the car, once. A failed fetch may be tried again. */
  function load() {
    if (!loading) {
      loading = import('./car_rig.js').then((m) => { mod = m; return true; });
      loading.catch(() => { loading = null; });
    }
    return loading;
  }

  return {
    load,
    ready: () => !!mod,
    // The chase camera and the grain on the ground are the first bike's
    // numbers, so that the three vehicles sit in the same picture.
    makeRig: (canvas) => mod.makeRig(canvas, Bike.TUNE),
    /** FÖHN's yaw turns from east towards north: a bearing b is pi/2 - b. */
    yawFor: (bearingDeg) => Math.PI / 2 - bearingDeg * Math.PI / 180,
  };
})();
