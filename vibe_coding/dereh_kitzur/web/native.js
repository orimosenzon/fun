/* Native - what changes when this page runs inside the Android app.
 *
 * The app (android_app/, Capacitor) is a window onto this very site, loaded
 * live from GitHub Pages, so a fix here reaches it without a new install.
 * What the app adds is reached through window.Capacitor, which only exists
 * inside it; in a browser every function here is a no-op and `isApp` is false.
 *
 * Two things live here:
 *   the walk   recorded by a native watcher behind a notification, so it goes
 *              on with the screen off and the phone in a pocket. A web page
 *              cannot do that: a locked screen stops it hearing the GPS.
 *   back       Android's back button closes what is open, the way Escape does
 *              on a desktop, and only leaves the app from the bare map.
 */
'use strict';

const Native = (() => {

  const cap = window.Capacitor;
  const isApp = !!(cap && cap.isNativePlatform && cap.isNativePlatform());
  const plugin = (name) => (isApp && cap.Plugins && cap.Plugins[name]) || null;

  /** Start hearing the GPS for a walk. `onFix` gets a position shaped like the
   *  browser's own (coords.latitude/longitude/accuracy, timestamp) so the
   *  recorder does not care which one it came from. Resolves to a stop(). */
  async function watchWalk(onFix, onError) {
    const geo = plugin('BackgroundGeolocation');
    if (!geo) throw new Error('no native geolocation');
    // Android 13+ hides a foreground service's notification unless this is
    // granted. The walk records either way; the notification is how the
    // walker knows it still does, and how they get back to it.
    const notes = plugin('LocalNotifications');
    if (notes) {
      try { await notes.requestPermissions(); } catch (err) { /* records anyway */ }
    }
    // Asked here, before the watcher, and not left to addWatcher: the plugin
    // starts its foreground service at once, Android refuses a location
    // service to an app that has no location permission yet, and the plugin
    // does not try again once the permission comes. Seen on the emulator on
    // 12/10/2026: points recorded, but no notification and no service, so
    // the walk would have stopped with the screen.
    let perm = await geo.checkPermissions().catch(() => null);
    if (!perm || perm.location !== 'granted') perm = await geo.requestPermissions();
    if (!perm || perm.location !== 'granted') {
      onError({ code: 'NOT_AUTHORIZED', message: 'location permission denied' });
      return () => {};
    }
    const id = await geo.addWatcher({
      backgroundTitle: 'דרך קיצור מקליטה שביל',
      backgroundMessage: 'ההקלטה ממשיכה גם כשהמסך כבוי. לחץ כדי לחזור.',
      requestPermissions: true,
      stale: false,
      distanceFilter: 0
    }, (loc, err) => {
      if (err) { onError(err); return; }
      if (!loc) return;
      onFix({
        coords: { latitude: loc.latitude, longitude: loc.longitude, accuracy: loc.accuracy },
        timestamp: loc.time || Date.now()
      });
    });
    return () => geo.removeWatcher({ id }).catch(() => {});
  }

  /** The app's own page in the phone's settings, where its location
   *  permission (and the "precise" switch) lives. */
  function openSettings() {
    const geo = plugin('BackgroundGeolocation');
    if (geo) geo.openSettings();
  }

  /* ---------- the back button ---------- */

  /* What is open, as one string: if pressing Escape changed it, Escape took
   * care of the back press. */
  function openState() {
    const shown = [...document.querySelectorAll('.sheet, .dk-modal, #lightbox, [role="dialog"]')]
      .filter((x) => !x.hidden && x.offsetParent !== null).length;
    return [document.body.className, shown,
      document.querySelectorAll('[aria-expanded="true"]').length,
      typeof selectedId !== 'undefined' ? selectedId : ''].join('|');
  }

  function onBack() {
    // A box drawn by this app over everything (the GPS log, the precise
    // location note) goes first.
    const modals = document.querySelectorAll('.dk-modal');
    if (modals.length) { modals[modals.length - 1].remove(); return; }

    // An open sheet closes the way a tap outside it does.
    const sheets = [...document.querySelectorAll('.sheet')].filter((s) => !s.hidden);
    if (sheets.length) {
      const top = sheets[sheets.length - 1];
      const x = top.querySelector('[data-act="close"]');
      (x || top).click();
      return;
    }

    // A recording in progress is never thrown away by a back press: the app
    // steps aside and the walk goes on behind its notification.
    const app = plugin('App');
    if (typeof Drafts !== 'undefined' && Drafts.isDrafting()) {
      if (app) app.minimizeApp();
      return;
    }

    // Everything else that Escape already knows how to back out of.
    const before = openState();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', code: 'Escape', bubbles: true }));
    if (openState() !== before) return;

    if (typeof selectedId !== 'undefined' && selectedId && typeof deselect === 'function') {
      deselect();
      return;
    }
    // The bare map: leave, but as Home would, so coming back finds it as it was.
    if (app) app.minimizeApp();
  }

  function wire() {
    const app = plugin('App');
    if (app) app.addListener('backButton', onBack);
    if (isApp) document.documentElement.classList.add('in-app');
  }

  wire();

  return { isApp, watchWalk, openSettings };
})();
