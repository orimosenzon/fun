/* The F-5 - the second jet bike of the flight mode (20/11/2026), for whoever
 * swaps to it; the first, bike.js, stays the default: see explore.js.
 *
 * This is the door to it and nothing more. The bike is FÖHN's, an ES module
 * (f5.js) that imports FÖHN's own files, so it is fetched on the first
 * take-off and not on page load, the way bike.js fetches Three.js. The
 * handles are bike.js's, so explore.js can hold either.
 */
'use strict';

const Bike5 = (() => {
  let mod = null;
  let loading = null;

  /** Fetch the bike, once. A failed fetch may be tried again. */
  function load() {
    if (!loading) {
      loading = import('./f5.js').then((m) => { mod = m; return true; });
      loading.catch(() => { loading = null; });
    }
    return loading;
  }

  return {
    load,
    ready: () => !!mod,
    // The chase camera and the grain on the ground are the first bike's
    // numbers, so that swapping bikes does not move the picture.
    makeRig: (canvas) => mod.makeRig(canvas, Bike.TUNE),
    /** FÖHN's yaw turns from east towards north: a bearing b is pi/2 - b. */
    yawFor: (bearingDeg) => Math.PI / 2 - bearingDeg * Math.PI / 180,
  };
})();
