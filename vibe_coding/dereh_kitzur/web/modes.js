/* View presets: הולך רגל, מתכנן, שקיפות (Ori, 3/10/2026).
 *
 * The app grew to two dozen layers, and three kinds of visitor came with them.
 * Somebody walking wants the shortcuts and nothing on top of them. Somebody
 * reading the town as a planner wants the land use, the plans in progress and
 * the cycling network. Somebody following what is being decided wants the
 * parcels glowing where an objection can still be filed. Each of those was
 * five or six ticks across three folded categories, and a first visitor had
 * no way to know any of it was there.
 *
 * A preset is just a set of layers, plus a nudge to the camera and the panel.
 * Which one is lit is worked out from the layers that are on rather than kept
 * as state of its own, so a link, a hand-ticked layer or the "clear all"
 * button can never leave a chip lit over a map it no longer describes: tick
 * one more box and you are in no preset, which is the truth.
 */
'use strict';

const Modes = (() => {
  const PRESETS = [
    {
      id: 'walk', icon: '🚶', name: 'הולך רגל',
      hint: 'קיצורי הדרך, השביל הסובב והמקומות שבדרך. בלי שום דבר מעליהם.',
      layers: () => [Layers.TRAILS_ID, 'kitzur-spots', Layers.SOVEV_ID],
      tilt: true, legend: false
    },
    {
      id: 'plan', icon: '📐', name: 'מתכנן',
      hint: 'ייעודי הקרקע בצבעי מבא"ת, התכניות בתהליך ורשת האופניים, על מפה שטוחה.',
      layers: () => [Layers.TRAILS_ID, Layers.LANDUSE_ID, Layers.PLANS_ID,
                     'bike-existing', 'bike-proposed'],
      tilt: false, legend: true
    },
    {
      id: 'open', icon: '🔍', name: 'שקיפות',
      hint: 'החלקות שאפשר להגיש עליהן התנגדות עכשיו, באדום, והתכניות שבתהליך.',
      layers: () => [Layers.TRAILS_ID, Layers.PARCELS_ID, Layers.BLOCKS_ID, Layers.PLANS_ID],
      tilt: false, legend: false
    }
  ];

  const same = (a, b) => a.length === b.length && a.every((x) => b.includes(x));

  /** The preset the map is showing right now, or null. Only layers that exist
   *  count: a preset naming one that failed to load still matches. */
  function current() {
    const on = Layers.onIds();
    return PRESETS.find((p) => same(p.layers().filter((id) => Layers.byId(id)), on)) || null;
  }

  function apply(id) {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    Layers.setOnly(p.layers());
    Layers.setLegendOpen(p.legend);
    // A plan is read flat, the way it is drawn; a walk is the tilted view the
    // app opens in. TILTED lives in app.js, read here at click time.
    const pitch = p.tilt ? TILTED : 0;
    if (map && Math.abs(map.getPitch() - pitch) > 3) {
      map.easeTo({ pitch, duration: 700 });
      el('tilt').classList.toggle('on', pitch >= 10);
    }
    paint();
    if (typeof scheduleSync === 'function') scheduleSync();
  }

  const chips = () => PRESETS.map((p) =>
    `<button class="mode" data-mode="${p.id}" title="${escapeHtml(p.hint)}">
       <span aria-hidden="true">${p.icon}</span> ${escapeHtml(p.name)}</button>`).join('');

  /** Light the chip of the preset that is showing, wherever chips are drawn,
   *  and tell the panel which cards belong to it (see .mode-* in app.css). */
  function paint() {
    const p = current();
    document.querySelectorAll('[data-modes]').forEach((box) => {
      if (!box.children.length) box.innerHTML = chips();
      box.querySelectorAll('.mode').forEach((b) => {
        const on = !!p && b.dataset.mode === p.id;
        b.classList.toggle('on', on);
        b.setAttribute('aria-pressed', on);
      });
    });
    const hint = el('mode-hint');
    if (hint) hint.textContent = p ? p.hint : 'תצוגה משלך. לחיצה על אחד המצבים מחזירה אליו.';
    PRESETS.forEach((x) => document.body.classList.toggle(`mode-${x.id}`, !!p && p.id === x.id));
  }

  function wire() {
    document.addEventListener('click', (e) => {
      const b = e.target.closest('[data-modes] .mode');
      if (b) apply(b.dataset.mode);
    });
    paint();
  }

  return { wire, paint, apply, current, PRESETS };
})();
