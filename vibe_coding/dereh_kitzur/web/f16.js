/* The F-16 - the door to it and nothing more, as bike5.js is the F-5's and
 * car.js the car's (26/11/2026).
 *
 * The aircraft is three ES modules: the flight (f16_core.js), the drawing of
 * its glass and dials (f16_draw.js) and the cockpit built around the pilot
 * (f16_pit.js, which needs three.js). They are fetched on the first take-off
 * and not on page load: nobody who came for a trail should pay for a
 * fighter they may never fly.
 */
'use strict';

const Viper = (() => {
  let mod = null;
  let loading = null;

  /** Fetch the aircraft, once. A failed fetch may be tried again. */
  function load() {
    if (!loading) {
      loading = import('./f16_pit.js').then((m) => { mod = m; return true; });
      loading.catch(() => { loading = null; });
    }
    return loading;
  }

  return { load, ready: () => !!mod, get mod() { return mod; } };
})();
