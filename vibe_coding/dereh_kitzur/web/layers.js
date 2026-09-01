/* Layers - the map is no longer one dataset.
 *
 * Four kinds of layer share one registry:
 *   trails   the initiative's own shortcuts, plus any further trail layer an
 *            editor has made. A trail carries the id of its layer; the ones
 *            that carry none belong to the initiative's original layer.
 *   network  the moshava's cycling-network plan, existing and proposed
 *   places   points somebody else already wrote up, each carrying that source's
 *            own words, photos and a link back. Two of them: the pardespedia
 *            articles, and the אמנות במושבה festival map.
 *   drafts   trails recorded on this device, held by draft.js
 *
 * Everything downstream - the list, the search, the detail pane, navigation -
 * reads from `visibleSegments()` and `visibleWaypoints()`, so turning a layer
 * off removes it from the whole app rather than only from the map.
 *
 * These files are plain scripts with no bundler, so they share one global
 * scope: `map`, `select`, `escapeHtml` and friends come from app.js, which is
 * loaded last and calls in here only after its data has arrived.
 */
'use strict';

const Layers = (() => {

  const list = [];                    // draw order, bottom first
  const index = new Map();            // segment/waypoint id -> {layer, item}
  const feature = new Map();          // item id -> {src, fid} for feature-state
  const shapeFeature = new Map();     // the same, for the layers drawn as areas
  const wired = new Set();            // layer ids whose map handlers are bound

  let onChange = () => {};            // set by app.js, re-renders the list

  const byId = (id) => list.find((l) => l.id === id);
  const item = (id) => (index.get(id) || {}).item;
  const layerOf = (id) => (index.get(id) || {}).layer;

  const TRAILS_ID = 'trails';
  const PLACES_ID = 'places';
  const PENDING_ID = 'pending';
  const ART_ID = 'art2026';
  const SHIMUR_ID = 'shimur';
  const MAKOM_ID = 'makom-shamur';
  const PLANS_ID = 'plans';
  const BLOCKS_ID = 'blocks';

  /* The circular route around the moshava, imported from off-road.io. Unlike
   * the rest of that file it is not a plan on paper but a marked route people
   * walk, so it belongs in the opening view rather than behind a toggle. */
  const SOVEV_ID = 'offroad-6177368023236608';

  /* Stored per browser, so a visitor who turns the cycling plan on keeps it on.
   * Only the on/off flags are stored - never the data itself, which is rebuilt
   * from the JSON on every load.
   *
   * The version in the key is how a change of defaults reaches somebody who has
   * been here before: savePrefs writes a flag for every layer at once, so an
   * earlier visitor carries an explicit `false` for a layer that has since
   * become one of the defaults and would never see the change. Bumped 29/8/2026,
   * when the circular route joined the opening view. */
  const PREF = 'dk.layers.v2';

  function loadPrefs() {
    try {
      return JSON.parse(localStorage.getItem(PREF) || '{}');
    } catch (err) {
      return {};
    }
  }

  /* A link is the second place a layer choice lives. The address bar always
   * names the layers currently switched on, so copying whatever is up there and
   * sending it to somebody opens the map on that exact combination instead of
   * on their own - which is the only way to say "look at the shortcuts together
   * with the festival" in a message.
   *
   * Only the layers everybody has are named. The drafts on this device and the
   * review queue mean nothing to whoever opens the link, and a link that
   * switched somebody's own recordings off would be a nasty surprise. */
  const URL_KEY = 'layers';
  const shareable = (l) => l.kind === 'trails' || l.kind === 'network'
    || l.kind === 'places' || l.kind === 'trips' || l.kind === 'waypoints';

  /** The layers a link asks for, or null when the URL says nothing about them -
   *  which is the difference between "show none of these" and "use whatever
   *  this browser chose last time". */
  function urlPrefs() {
    const raw = new URLSearchParams(location.search).get(URL_KEY);
    return raw === null ? null : new Set(raw.split(',').filter(Boolean));
  }

  /** Put the current choice back in the address bar, replacing the entry rather
   *  than adding one: the back button belongs to the map, not to a checkbox. */
  function syncUrl() {
    const params = new URLSearchParams(location.search);
    params.set(URL_KEY, list.filter((l) => l.on && shareable(l))
      .map((l) => l.id).join(','));
    // A comma is legal in a query string and URLSearchParams escapes it anyway,
    // which turns a link somebody is about to paste into a message into a wall
    // of %2C. Ours are the only commas here, so it is safe to put them back.
    const query = params.toString().replace(/%2C/g, ',');
    try {
      history.replaceState(null, '', `${location.pathname}?${query}${location.hash}`);
    } catch (err) {
      /* opened straight off the filesystem; the toggles still work */
    }
  }

  /** Both places the choice is kept: this browser, and the link. */
  function savePrefs() {
    const on = {};
    list.forEach((l) => { on[l.id] = l.on; });
    try {
      localStorage.setItem(PREF, JSON.stringify(on));
    } catch (err) {
      /* private mode: the toggles simply do not persist */
    }
    syncUrl();
  }

  /* ---------- registry ---------- */

  function add(layer) {
    layer.segments = layer.segments || [];
    layer.waypoints = layer.waypoints || [];
    list.push(layer);
    return layer;
  }

  function reindex() {
    index.clear();
    list.forEach((layer) => {
      layer.segments.forEach((s) => index.set(s.id, { layer, item: s }));
      layer.waypoints.forEach((w) => index.set(w.id, { layer, item: w }));
    });
  }

  /** An article or a festival entry turned into something the list and the
   *  detail pane can treat like any other point on the map.
   *
   *  The position lives under `geo` in the file, because build_places.py has to
   *  tell a hand-dropped pin from one it guessed off an address, and only ever
   *  overwrite the guesses. The rest of the app just wants lat and lng. */
  function toPlace(raw, colours, fallback) {
    const geo = raw.geo || null;
    return {
      ...raw,
      lat: geo ? geo.lat : null,
      lng: geo ? geo.lng : null,
      unplaced: !geo,
      // `spread` means the builder nudged this off a pile it shared with its
      // neighbours so that all of them stay clickable. The pin is then a few
      // metres from where its source put it, and saying so is the honest thing:
      // "מיקום מקורב" is exactly what it now is.
      approx: !!geo && (geo.source === 'nearby' || geo.source === 'street' || !!geo.spread),
      // A narrower claim than `approx`, and the one the map draws. `spread`
      // moves a pin a few metres and covers half of some layers, so drawing on
      // it would empty out the festival and most of pardespedia over what is a
      // drawing device. These four are the sources that do not know where the
      // place is at all: a street with no house number, a spot mentioned
      // alongside it, a neighbouring parcel. That is worth showing.
      vague: !!geo && ['nearby', 'street', 'neighbour'].includes(geo.source),
      geoSource: geo ? geo.source : null,
      place: true,
      // A source that sorts its places into groups colours them by group. One
      // that does not - the cadastral blocks are all the same kind of thing -
      // takes the layer's own colour, and the purple is the last resort for a
      // place whose group is not in the document's list at all.
      color: colours[raw.group] || fallback || '#6a1b9a'
    };
  }

  /** One layer of places, from a document shaped like places.json.
   *
   *  Two of these exist and they differ in one way that matters: whose map it
   *  is. The pardespedia places have no coordinates in the wiki at all, so this
   *  app is where their positions are decided and an editor may drag them. The
   *  festival placed its own pins on its own map, and those are not ours to
   *  move - hence `pinnable`, which gates every editing affordance the detail
   *  pane and the layer sheet offer for places.
   *
   *  Each caller also names its source twice over: `linkTitle` for the link at
   *  the head of the detail pane, `sourceLine` for the credit at the foot. */
  function addPlaceLayer(id, doc, opts) {
    if (!doc || !doc.places || !doc.places.length) return null;
    const colours = {};
    (doc.groups || []).forEach((g) => { colours[g.name] = g.color; });
    return add({
      ...opts,
      id,
      kind: 'places',
      pinnable: !!opts.pinnable,
      groups: doc.groups || [],
      waypoints: doc.places.map((p) => toPlace(p, colours, opts.color))
    });
  }

  /* ---------- trips ----------
   *
   * A trip is a continuous route through the moshava that chains shortcuts
   * already on the map together with pieces drawn between them. It is stored
   * as what it is made of and never as what it looks like:
   *
   *     parts: [ {trail:'p52', reversed:true}, {draw:[[lat,lng],…]}, {trail:'p13'} ]
   *
   * so that correcting a shortcut's geometry corrects every trip that walks
   * through it, and so that a trip can say which shortcuts it uses and a
   * shortcut can say which trips pass along it. The price is that a shortcut
   * somebody deletes leaves a hole in every trip built on it, and a hole has
   * to be visible rather than quietly closed with a straight line - hence
   * `missing`, which the detail pane says out loud.
   */

  const TRIPS_ID = 'trips';
  const SPOTS_ID = 'kitzur-spots';

  /** How far apart two ends may be and still count as the same place. Below
   *  this it is the imprecision of where somebody stopped recording; above it,
   *  it is ground nobody has walked, and the trip editor makes you draw it. */
  const TRIP_GAP_M = 25;

  /** Whether a walk brings you back to where you left the car, which is the
   *  first thing somebody choosing one wants to know.
   *
   *  The threshold has to scale with the walk or it says silly things at both
   *  ends: a flat 150 m calls a 165 m stroll circular because its ends happen
   *  to be near, and would call a twelve-kilometre round trip point-to-point
   *  over a couple of streets. A quarter of the distance, capped, is the rule -
   *  "the ends are close compared to how far you walked". */
  const LOOP_M = 150;
  const isLoop = (path, length) => path.length > 1
    && metres(path[0], path[path.length - 1])
       <= Math.min(LOOP_M, (length != null ? length : pathLength(path)) / 4);

  const DIFFICULTY = [
    { name: 'קל', color: '#2e7d32' },
    { name: 'בינוני', color: '#ef6c00' },
    { name: 'מאתגר', color: '#c62828' }
  ];

  const metres = (a, b) => distance({ lat: a[0], lng: a[1] }, { lat: b[0], lng: b[1] });
  const same = (a, b) => Math.abs(a[0] - b[0]) < 1e-6 && Math.abs(a[1] - b[1]) < 1e-6;

  function pathLength(path) {
    let sum = 0;
    for (let i = 1; i < path.length; i++) sum += metres(path[i - 1], path[i]);
    return Math.round(sum);
  }

  /** A recipe turned into a line, plus what it turned out to be made of.
   *
   *  `lookup` is passed in because this runs against three different sets of
   *  trails: the published ones for a trip on the map, the same for a draft on
   *  this device, and the document being written for one going through the
   *  queue. */
  function resolveTrip(parts, lookup) {
    const find = lookup || ((id) => {
      const it = item(id);
      return it && it.path ? it : null;
    });
    const path = [];
    const uses = [];
    const missing = [];

    (parts || []).forEach((part) => {
      let piece;
      if (part.trail) {
        const seg = find(part.trail);
        if (!seg || !seg.path || seg.path.length < 2) { missing.push(part.trail); return; }
        piece = part.reversed ? seg.path.slice().reverse() : seg.path;
        uses.push({ id: part.trail, name: seg.name, reversed: !!part.reversed });
      } else {
        piece = part.draw || [];
      }
      if (!piece.length) return;
      // Two parts that meet at a point would otherwise store it twice: harmless
      // on the map, wrong in every count and in the length.
      const last = path[path.length - 1];
      const from = last && same(last, piece[0]) ? 1 : 0;
      for (let i = from; i < piece.length; i++) path.push(piece[i]);
    });

    return { path, uses, missing };
  }

  /** One stored trip, presented the way the rest of the app expects a segment. */
  function toTrip(raw, lookup) {
    const { path, uses, missing } = resolveTrip(raw.parts, lookup);
    const length = pathLength(path);
    const colours = {};
    DIFFICULTY.forEach((d) => { colours[d.name] = d.color; });
    return {
      ...raw,
      path,
      uses,
      missing,
      length,
      trip: true,
      group: raw.difficulty || '',
      // A number the walker can plan around. Four km/h is a flat-ground pace
      // and the moshava is flat; anybody who knows better types their own.
      minutes: raw.minutes || Math.round((length / 1000) * 15),
      loop: isLoop(path, length),
      color: raw.color || colours[raw.difficulty] || '#00695c',
      entries: path.length > 1
        ? [{ lat: path[0][0], lng: path[0][1] },
           { lat: path[path.length - 1][0], lng: path[path.length - 1][1] }]
        : []
    };
  }

  /** Every trip in the trails document, resolved against that same document's
   *  shortcuts rather than against the index, which is not built yet. */
  function buildTrips(trails) {
    const segs = new Map((trails.segments || []).map((s) => [s.id, s]));
    return (trails.trips || [])
      .map((t) => toTrip(t, (id) => segs.get(id)))
      .filter((t) => t.path.length > 1);
  }

  /** The map layers a tap can land on and mean "this shortcut" - what the trip
   *  editor asks before deciding a tap was a point it should draw. Drafts are
   *  left out: a trip may only be built out of what is actually on the map. */
  const trailHitLayers = () => list
    .filter((l) => l.kind === 'trails' && l.on && l.id !== 'drafts')
    .map((l) => hitId(l.id))
    .filter((id) => typeof map !== 'undefined' && map && map.getLayer(id));

  /** Switch a layer on from code, as the trip editor does with the shortcuts:
   *  you cannot chain what you cannot see. */
  function turnOn(id) {
    const layer = byId(id);
    if (!layer || layer.on) return false;
    layer.on = true;
    savePrefs();
    applyVisibility();
    onChange();
    return true;
  }

  /** The trips that walk along one shortcut, for its detail pane. Deleting a
   *  shortcut is not a local act once a trip is built on it. */
  const tripsUsing = (trailId) => {
    const layer = byId(TRIPS_ID);
    return layer ? layer.segments.filter(
      (t) => (t.uses || []).some((u) => u.id === trailId)) : [];
  };

  /** The trail layers, rebuilt from the document.
   *
   *  `on` comes from a lookup rather than a stored map so the same code serves
   *  the first load, where it reads the browser's preferences, and a reload
   *  after a write, where it reads whatever is currently switched on. */
  function buildTrailLayers(trails, isOn) {
    const custom = trails.layers || [];
    const mine = new Set(custom.map((l) => l.id));

    // A trail whose layer was deleted, or which names a layer this app has
    // never heard of, belongs with the initiative's own rather than nowhere.
    const home = (it) => (it.layer && mine.has(it.layer) ? it.layer : TRAILS_ID);

    add({
      id: TRAILS_ID,
      kind: 'trails',
      category: 'trails',
      name: 'דרכי קיצור',
      short: 'קיצור',
      // One colour for all of them. They arrived from My Maps in seven, which
      // was that map's own sorting and meant nothing here, and a shortcut is a
      // shortcut. A trail may still be given a colour of its own from its page,
      // and that stays an exception rather than the rule.
      color: '#4a148c',
      note: 'קיצורי הדרך שמופו על ידי יוזמת דרך קיצור.',
      source: trails.source,
      on: isOn(TRAILS_ID),                 // the point of the app; on unless muted
      segments: trails.segments.filter((s) => home(s) === TRAILS_ID),
      waypoints: []                        // they have their own layer, below
    });

    // The initiative's own waypoints, which are not shortcuts at all: a tree, a
    // shelter, a bench, a gap in a fence. They sat inside the shortcuts layer
    // because they arrived in the same file, and that made "show me the
    // shortcuts" mean "show me the shortcuts and eleven other things". Their own
    // layer, switched on by default because that is where they already were.
    add({
      id: SPOTS_ID,
      kind: 'waypoints',
      category: 'places',
      name: 'מקומות של דרך קיצור',
      short: 'מקום קיצור',
      unit: 'מקומות',
      color: '#8d6e63',                    // the colour the pins are drawn in
      note: 'נקודות שהיוזמה סימנה בדרך: עצים, מקלטים, ספסלים, מעברים חסומים. '
        + 'לא שבילים, ולכן לא בשכבת דרכי הקיצור.',
      on: isOn(SPOTS_ID, true),
      segments: [],
      waypoints: trails.waypoints.filter((w) => home(w) === TRAILS_ID)
    });

    custom.forEach((l) => add({
      ...l,
      kind: 'trails',
      category: 'trails',
      own: true,                           // made in the app, so it can be edited here
      on: isOn(l.id),
      segments: trails.segments.filter((s) => home(s) === l.id),
      waypoints: trails.waypoints.filter((w) => home(w) === l.id)
    }));
  }

  /* What a visitor sees on arrival, before touching anything: the initiative's
   * own shortcuts and the circular route. The cycling plan and the several
   * hundred pardespedia pins are a tap away in the layer sheet, and putting
   * them all on the map at once buries the shortcuts under them. */
  function init(trails, network, places, art, shimur, makom, plans, blocks) {
    const prefs = loadPrefs();
    const link = urlPrefs();
    /** What the link asks for, else what this browser chose, else the default
     *  for a layer nobody has touched.
     *
     *  A link names its layers in full, so it decides both halves: a layer it
     *  leaves out is off even if this browser had it on. `fromLink` is false
     *  for the layers that live on this device only, which no link may touch. */
    const isOn = (id, byDefault, fromLink = true) => (
      link && fromLink ? link.has(id)
        : typeof prefs[id] === 'boolean' ? prefs[id] : byDefault);

    // The shortcuts are the point of the app, and a trail layer an editor makes
    // later is the same content sorted into buckets, so both arrive on.
    buildTrailLayers(trails, (id) => isOn(id, true));

    // Walks made of those shortcuts. Off by default like everything else that
    // is not a shortcut, and drawn under them as a wide band rather than over,
    // so that turning it on shows you which shortcuts a walk threads together
    // instead of covering them up.
    add({
      id: TRIPS_ID,
      kind: 'trips',
      category: 'trails',
      name: 'טיולים',
      short: 'טיול',
      unit: 'טיולים',
      color: '#00695c',
      groups: DIFFICULTY,
      note: 'מסלולי הליכה רציפים, שרובם משרשרים קיצורי דרך שכבר על המפה עם '
        + 'קטעים מצוירים ביניהם. הצבע לפי דרגת הקושי. כדי להוסיף טיול צריך '
        + 'ששכבת דרכי הקיצור תהיה דלוקה, כי משרשרים בלחיצה עליהן.',
      on: isOn(TRIPS_ID, false),
      segments: buildTrips(trails)
    });

    (network ? network.layers : []).forEach((l) => add({
      ...l,
      kind: 'network',
      category: 'trails',
      on: isOn(l.id, l.id === SOVEV_ID)    // the rest is planning data, opt-in
    }));

    addPlaceLayer(PLACES_ID, places, {
      name: 'מקומות מפרדספדיה',
      category: 'places',
      short: 'מקום',
      color: '#7b1fa2',
      note: 'בתי קפה, גנים, מוסדות ואתרי הנצחה, עם התקציר והתמונה מהערך בוויקי.',
      credit: 'pardespedia.info',
      sourceName: 'פרדספדיה, הוויקי של המושבה',
      sourceLine: 'הטקסט והתמונה מתוך פרדספדיה, הוויקי של המושבה',
      linkTitle: 'הערך המלא בפרדספדיה',
      pinnable: true,                      // the wiki holds no coordinates
      on: isOn(PLACES_ID, false)           // hundreds of pins; opt-in
    });

    // The festival's own map, read once a year by build_art.py. It is a
    // fortnight in the life of the moshava rather than part of its furniture,
    // so it is off until somebody asks for it - but everything in it is within
    // walking distance of everything else, which is the whole point of pairing
    // it with the shortcuts.
    addPlaceLayer(ART_ID, art, {
      name: (art && art.name) || 'אמנות במושבה 2026',
      category: 'places',
      short: 'אמנות',
      color: '#c2185b',
      note: 'סטודיואים פתוחים, תערוכות, מוזיקה ואוכל בפסטיבל אמנות במושבה. '
        + 'הטקסט, התמונות ופרטי הקשר מגיעים מהמפה של הפסטיבל.',
      credit: 'אמנות במושבה · pardesart.co.il',
      sourceName: 'אמנות במושבה',
      sourceLine: 'הטקסט והתמונות מתוך המפה של אמנות במושבה',
      linkTitle: 'הדף המלא באתר הפסטיבל',
      pinnable: false,                     // the festival placed these itself
      on: isOn(ART_ID, false)
    });

    // What the moshava has decided is worth keeping. Two lists rather than
    // one, because they do not mean the same thing: the appendix is the law
    // and מקום שמור is an argument, and merging them would quietly promote
    // somebody's proposal into a statutory grade. Both are off by default -
    // walking routes are what the app is for - and both pair naturally with
    // the shortcuts, which is how you end up walking past a water tower.
    addPlaceLayer(SHIMUR_ID, shimur, {
      name: 'אתרים לשימור: תוכנית המתאר',
      category: 'places',
      short: 'שימור',
      color: '#ef6c00',
      note: 'האתרים שנקבעו לשימור בנספח השימור של תוכנית המתאר הכוללנית: '
        + 'מבנים, מגדלי מים, שדרות עצים ומתחמים היסטוריים. לכל אתר דרגת השימור '
        + 'שנקבעה לו. הצבעים לפי הנרטיב שאליו האתר משויך בנספח. '
        + 'הנספח מוסר גוש וחלקה ולא נקודה, ולכן כל נקודה כאן היא מרכז החלקה '
        + 'ולא המבנה עצמו.',
      credit: 'נספח השימור, תכנית 353-0138586',
      sourceName: 'נספח השימור של תוכנית המתאר הכוללנית',
      sourceLine: 'מתוך נספח השימור של תוכנית המתאר הכוללנית (ד"ר הדס שדר, 2017)',
      linkTitle: 'נספח השימור המלא באתר מנהל התכנון',
      pinnable: false,                     // positions come off the cadastre
      on: isOn(SHIMUR_ID, false)
    });

    addPlaceLayer(MAKOM_ID, makom, {
      name: 'מקום שמור',
      category: 'places',
      short: 'מקום שמור',
      color: '#ad1457',
      note: 'רשימת האתרים של פרויקט מקום שמור, פרויקט התיעוד של אילנה פלדה. '
        + 'הרשימה היא הצעה ואין לה מעמד סטטוטורי, והיא כוללת גם מבנים שלא נכללו '
        + 'בנספח השימור. רוב האתרים עוד לא מוקמו. התמונות מגיעות מהערכים של '
        + 'הפרויקט עצמו, והכיתוב שמתחת לכל אחת הוא הכיתוב שהפרויקט כתב לה, '
        + 'עם הצלם והארכיון שממנו היא הגיעה.',
      credit: 'מקום שמור · makomshamur.com',
      sourceName: 'מקום שמור',
      sourceLine: 'מתוך פרויקט מקום שמור, אילנה פלדה',
      linkTitle: 'הדף באתר מקום שמור',
      pinnable: false,                     // no arrange tool for this one yet
      on: isOn(MAKOM_ID, false)
    });

    // What somebody is trying to build here. Unlike everything else on this map
    // it is about the future rather than the ground, and unlike the conservation
    // layers it changes every few weeks - a plan moves from one stage to the
    // next, and the fortnight it spends open to objections is the only stretch
    // when a resident can do anything about it. Drawn as areas, because a plan
    // is an area: the blue line is most of what it says.
    addPlaceLayer(PLANS_ID, plans, {
      name: 'תכניות בתהליך',
      category: 'other',
      short: 'תכנית',
      unit: 'תכניות',
      labels: false,                       // see addPlaceLayers
      color: '#01579b',
      note: 'תכניות בנייה שנמצאות בהליך תכנוני בפרדס חנה-כרכור, מהגשה ועד '
        + 'אישור, לפי מאגר מנהל התכנון. הצבע לפי השלב, והכהה שבהם הוא ההפקדה, '
        + 'התקופה שבה אפשר להגיש התנגדות. הגבול הוא הקו הכחול של התכנית. '
        + 'תכניות שכבר אושרו ותיקים שנסגרו אינם בשכבה הזאת.',
      credit: 'מנהל התכנון · Xplan',
      sourceName: 'מנהל התכנון',
      sourceLine: 'מתוך מאגר התכניות המקוונות של מנהל התכנון',
      linkTitle: 'התכנית המלאה במבא"ת',
      pinnable: false,                     // the boundary comes off the register
      on: isOn(PLANS_ID, false)
    });

    // The reference grid the planning layer's names are written in. Every plan
    // is called after a block and a parcel - "תוספת זכויות בניה בגוש 10105
    // חלקה 203" - and without this there is no way to find out from the map
    // which block you are standing in. Dashed, because a block boundary is a
    // line in a register and not a thing you can walk into.
    addPlaceLayer(BLOCKS_ID, blocks, {
      name: 'גושים',
      category: 'other',
      short: 'גוש',
      unit: 'גושים',
      color: '#8d6e63',
      dash: true,                          // drawn as a grid: outline, no fill
      dots: false,                         // the outline is the block, not a point
      note: 'גבולות הגושים של הקדסטר הארצי, עם מספר הגוש. שמות התכניות בשכבת '
        + 'התכניות מפנים לגוש ולחלקה, וזו הדרך למצוא אותם על הקרקע. '
        + 'החלקות עצמן אינן כאן: יש מעל עשרת אלפים מהן במושבה.',
      credit: 'הקדסטר הארצי · govmap',
      sourceName: 'הקדסטר הארצי',
      sourceLine: 'גבולות הגושים מתוך הקדסטר הארצי, govmap',
      linkTitle: 'govmap',
      pinnable: false,                     // the boundary comes off the cadastre
      on: isOn(BLOCKS_ID, false)
    });

    // Populated from the worker, and only while edit mode is on: a trail nobody
    // has looked at yet is not something to show a visitor as if it were part
    // of the map.
    add({
      id: PENDING_ID,
      kind: 'pending',
      category: 'trails',
      name: 'ממתינים לאישור',
      short: 'ממתין',
      color: '#f9a825',
      dash: true,
      note: 'תור: שבילים שתושבים שלחו דרך "שלח ליוזמה" וממתינים שתאשר אותם. '
        + 'שביל שאתה מפרסם בעצמך לא עובר דרך כאן. גלוי רק במצב עריכה.',
      on: true,
      segments: []
    });

    // Populated by draft.js once IndexedDB answers.
    add({
      id: 'drafts',
      kind: 'drafts',
      category: 'trails',
      name: 'הטיוטות שלי',
      short: 'טיוטות',
      color: '#8e24aa',
      dash: true,
      note: 'שבילים שהקלטת או ציירת במכשיר הזה. נשמרים כאן בלבד, עד שתשלח אותם.',
      // Somebody's own recording, on their own device. Hiding it by default
      // would mean walking a trail and not seeing it come out on the map, and
      // a link somebody else sent has no business hiding it either.
      on: isOn('drafts', true, false),
      segments: []
    });

    // The list is drawn bottom-first, but the plan should sit *under* the
    // trails on the map, so the map order is the reverse of the panel order.
    list.sort((a, b) => order(a) - order(b));
    reindex();
    // From here on the address bar is a truthful copy of what is on screen,
    // including on a first visit that arrived with a bare URL.
    syncUrl();

    const legendBtn = document.getElementById('legend-btn');
    if (legendBtn) legendBtn.addEventListener('click', toggleLegend);
    renderLegend();
  }

  const RANK = { network: 0, places: 1, trips: 1.5, trails: 2,
                 waypoints: 2.5, pending: 3, drafts: 4 };
  // The plans are a places layer that draws areas, and an area belongs under
  // every dot on the map rather than washing the colour out of the ones that
  // happen to fall inside it.
  const order = (l) => (l.id === BLOCKS_ID ? 0.25
    : l.id === PLANS_ID ? 0.5 : RANK[l.kind]);

  /** Rebuild the trail layers after a write, without disturbing the drafts
   *  layer or anyone's on/off choices. */
  function resetTrails(trails) {
    const was = {};
    list.forEach((l) => { if (l.kind === 'trails') was[l.id] = l.on; });

    const others = list.filter((l) => l.kind !== 'trails');
    list.length = 0;
    buildTrailLayers(trails, (id) => was[id] !== false);
    others.forEach((l) => list.push(l));

    // A trip is a recipe over these very shortcuts, so a write that moves one
    // of them moves every trip that walks along it. Rebuilding the lines here
    // is what makes "fix the shortcut, fix the trips" true.
    const trips = byId(TRIPS_ID);
    if (trips) {
      trips.segments = buildTrips(trails);
      if (typeof map !== 'undefined' && map && map.getSource(srcId(TRIPS_ID))) {
        map.getSource(srcId(TRIPS_ID)).setData(geojson(trips));
      }
    }
    list.sort((a, b) => order(a) - order(b));
    reindex();
    savePrefs();
    if (typeof map !== 'undefined' && map) addToMap();
    onChange();
  }

  /** Fill the review queue layer from what the worker holds. */
  function setPending(items) {
    const layer = byId(PENDING_ID);
    if (!layer) return;
    layer.segments = (items || [])
      .map((it) => (it.parts ? toTrip(it) : it))
      .filter((it) => it.path && it.path.length > 1)
      .map((it) => ({ ...it, color: '#f9a825', pending: true }));
    reindex();
    if (typeof map !== 'undefined' && map && map.getSource(srcId(PENDING_ID))) {
      map.getSource(srcId(PENDING_ID)).setData(geojson(layer));
    }
    applyVisibility();
    onChange();
  }

  /** Rebuild the places layer in place, after a pin is dropped or removed. */
  function resetPlaces(places) {
    const layer = byId(PLACES_ID);
    if (!layer || !places) return;
    const colours = {};
    (places.groups || []).forEach((g) => { colours[g.name] = g.color; });
    layer.waypoints = places.places.map((p) => toPlace(p, colours));
    reindex();
    if (typeof map !== 'undefined' && map && map.getSource(srcId(PLACES_ID))) {
      map.getSource(srcId(PLACES_ID)).setData(geojson(layer));
    }
    onChange();
  }

  /* ---------- what the rest of the app sees ---------- */

  const shown = (l) => l.on && !(l.kind === 'pending' && !Store.isEditor());
  const visible = () => list.filter(shown);
  const visibleSegments = () => visible().flatMap((l) => l.segments);
  const visibleWaypoints = () => visible().flatMap((l) => l.waypoints);

  /** Waypoints drawn as HTML markers, which is the initiative's own handful.
   *  Places are hundreds of points and go through a GL layer instead. */
  const markerWaypoints = () =>
    visible().filter((l) => l.kind !== 'places').flatMap((l) => l.waypoints);

  const trailLayers = () => list.filter((l) => l.kind === 'trails' && l.id !== 'drafts');

  function stats() {
    // The queue is not part of the map yet, so it gets its own number rather
    // than inflating the count of what is actually out there.
    const segs = visible().filter((l) => l.kind !== 'pending').flatMap((l) => l.segments);
    const waiting = visible().filter((l) => l.kind === 'pending')
      .reduce((n, l) => n + l.segments.length, 0);
    const wps = visible().filter((l) => l.kind !== 'places').flatMap((l) => l.waypoints);
    const places = visible().filter((l) => l.kind === 'places').flatMap((l) => l.waypoints);
    return {
      segments: segs.length,
      length: segs.reduce((sum, s) => sum + (s.length || 0), 0),
      waiting,
      waypoints: wps.length,
      places: places.length,
      unplaced: places.filter((p) => p.unplaced).length,
      photos: [...segs, ...wps, ...places]
        .reduce((sum, s) => sum + (s.photos ? s.photos.length : 0), 0)
    };
  }

  /* ---------- map ---------- */

  const srcId = (id) => `src-${id}`;
  const lineId = (id) => `ln-${id}`;
  const dotId = (id) => `pt-${id}`;
  const labelId = (id) => `lb-${id}`;
  const hitId = (id) => `hit-${id}`;
  const shapeSrc = (id) => `shp-${id}`;
  const fillId = (id) => `fl-${id}`;
  const edgeId = (id) => `eg-${id}`;

  /** Whether a places layer also carries an outline for each of its points.
   *  Only the planning schemes do: a plan is an area, and its boundary - the
   *  blue line - is most of what it says. */
  const hasShapes = (layer) => layer.kind === 'places'
    && layer.waypoints.some((p) => p.shape && p.shape.length);

  const drawnIds = (layer) => (layer.kind === 'waypoints' ? []
    : layer.kind === 'places'
    ? (hasShapes(layer) ? [fillId(layer.id), edgeId(layer.id)] : [])
      .concat([dotId(layer.id), labelId(layer.id), hitId(layer.id)])
    : [lineId(layer.id), hitId(layer.id)]);

  function geojson(layer) {
    if (layer.kind === 'places') {
      let i = 0;
      return {
        type: 'FeatureCollection',
        features: layer.waypoints.filter((p) => !p.unplaced).map((p) => {
          feature.set(p.id, { src: srcId(layer.id), fid: i });
          return {
            type: 'Feature',
            id: i++,
            properties: { id: p.id, color: p.color, name: p.name, vague: !!p.vague },
            geometry: { type: 'Point', coordinates: [p.lng, p.lat] }
          };
        })
      };
    }
    return {
      type: 'FeatureCollection',
      // A line needs two points. A trip whose shortcuts have all gone missing
      // has none, and it still belongs in the list and the detail pane - it is
      // only the map that cannot show it. Filtering here rather than upstream
      // is what keeps those two facts from fighting.
      features: layer.segments.filter((s) => s.path && s.path.length > 1).map((seg, i) => {
        feature.set(seg.id, { src: srcId(layer.id), fid: i });
        return {
          type: 'Feature',
          id: i,
          properties: { id: seg.id, color: seg.color || layer.color },
          geometry: {
            type: 'LineString',
            coordinates: seg.path.map(([lat, lng]) => [lng, lat])
          }
        };
      })
    };
  }

  /** The outlines of a layer that has them, as their own source.
   *
   *  A second source rather than polygons in the first, because the dots are
   *  drawn from `geojson` and a circle layer over a polygon source would draw
   *  nothing. The feature ids are kept in their own map for the same reason:
   *  one item is now two features in two sources, and highlighting has to reach
   *  both. */
  function shapeGeojson(layer) {
    let i = 0;
    return {
      type: 'FeatureCollection',
      features: layer.waypoints.filter((p) => p.shape && p.shape.length).map((p) => {
        shapeFeature.set(p.id, { src: shapeSrc(layer.id), fid: i });
        return {
          type: 'Feature',
          id: i++,
          properties: { id: p.id, color: p.color },
          geometry: { type: 'Polygon', coordinates: p.shape }
        };
      })
    };
  }

  function addShapeLayers(layer) {
    const src = shapeSrc(layer.id);
    const sel = ['boolean', ['feature-state', 'sel'], false];
    const dim = ['boolean', ['feature-state', 'dim'], false];

    if (map.getSource(src)) map.getSource(src).setData(shapeGeojson(layer));
    else map.addSource(src, { type: 'geojson', data: shapeGeojson(layer) });

    // A dashed area is a reference grid rather than something on the ground -
    // the cadastral blocks - and it draws as an outline with no fill at all,
    // because sixty of them tile the entire town and any fill would tint every
    // other layer through it. A solid one is content: a plan, faint enough that
    // ninety-five overlapping ones still leave the streets underneath readable.
    const grid = !!layer.dash;

    map.addLayer({
      id: fillId(layer.id),
      type: 'fill',
      source: src,
      paint: {
        'fill-color': ['get', 'color'],
        'fill-opacity': grid
          ? ['case', sel, 0.14, 0]
          : ['case', sel, 0.32, dim, 0.05, 0.14]
      }
    });

    map.addLayer({
      id: edgeId(layer.id),
      type: 'line',
      source: src,
      paint: Object.assign({
        'line-color': ['get', 'color'],
        'line-width': ['case', sel, 3.5, grid ? 1 : 1.6],
        'line-opacity': ['case', sel, 1, dim, 0.25, grid ? 0.5 : 0.8]
      }, grid ? { 'line-dasharray': [3, 2] } : {})
    });

    // Tapping anywhere inside a plan selects it, which is how you ask "what is
    // planned for my street" without having to find its centre dot.
    if (wired.has(fillId(layer.id))) return;
    wired.add(fillId(layer.id));
    map.on('click', fillId(layer.id), (e) => {
      if (e.features && e.features.length) select(e.features[0].properties.id, false);
    });
    map.on('mouseenter', fillId(layer.id), () => { map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', fillId(layer.id), () => { map.getCanvas().style.cursor = ''; });
  }

  function addPlaceLayers(layer) {
    const src = srcId(layer.id);
    const sel = ['boolean', ['feature-state', 'sel'], false];
    const dim = ['boolean', ['feature-state', 'dim'], false];
    // A point the builder could not put on its own spot - it went to a
    // neighbouring parcel, or to the street the address names. Drawn hollow:
    // pale fill, the group's colour moved out to the ring. Same hue, so it
    // still reads as its group, and visibly less certain than the rest.
    const soft = ['boolean', ['get', 'vague'], false];

    // A dot at the centre of a cadastral block means nothing - the block is its
    // outline, and the point is only there to hang the number on. The invisible
    // hit circle below stays, so the number is still something you can tap.
    if (layer.dots !== false) map.addLayer({
      id: dotId(layer.id),
      type: 'circle',
      source: src,
      paint: {
        'circle-color': ['get', 'color'],
        // A zoom expression has to be the outermost one: MapLibre rejects a
        // ["zoom"] nested inside a ["case"], so the selected state is decided
        // at each stop rather than around the whole thing.
        'circle-radius': ['interpolate', ['linear'], ['zoom'],
          12, ['case', sel, 7, 3.5],
          15, ['case', sel, 9, 5.5],
          18, ['case', sel, 12, 8]],
        'circle-opacity': ['case', dim, 0.45, soft, 0.3, 0.9],
        'circle-stroke-color': ['case', soft, ['get', 'color'], '#fff'],
        'circle-stroke-width': ['case', sel, 3, soft, 2, 1.5]
      }
    });

    // Hebrew labels need the style's glyph set to carry the Hebrew range.
    // OpenFreeMap's Noto Sans does, and the satellite style in app.js points
    // its `glyphs` at the same endpoint so both look the same.
    //
    // The planning layer asks for none, and it is right to. Its names are
    // sentences - "תוספת זכויות בניה ויח״ד בגוש 10105 חלקה 203, רח' הנדיב" -
    // and forty of them across the middle of the moshava bury the streets under
    // text that is nearly the same on every one. The outline says where, the
    // colour says which stage, and a tap says the rest.
    if (layer.labels !== false) map.addLayer({
      id: labelId(layer.id),
      type: 'symbol',
      source: src,
      minzoom: 14.5,
      layout: {
        'text-field': ['get', 'name'],
        'text-font': ['Noto Sans Regular'],
        'text-size': 12,
        'text-anchor': 'top',
        'text-offset': [0, 0.8],
        'text-max-width': 9,
        'text-padding': 4,
        'text-optional': true          // a label that will not fit is dropped,
      },                               // the dot stays
      paint: {
        'text-color': '#2b2b2b',
        'text-halo-color': 'rgba(255,255,255,0.9)',
        'text-halo-width': 1.6,
        'text-opacity': ['case', ['boolean', ['feature-state', 'dim'], false], 0.5, 1]
      }
    });

    map.addLayer({
      id: hitId(layer.id),
      type: 'circle',
      source: src,
      paint: { 'circle-color': '#000', 'circle-opacity': 0, 'circle-radius': 16 }
    });
  }

  function addLineLayers(layer) {
    const src = srcId(layer.id);
    const sel = ['boolean', ['feature-state', 'sel'], false];
    const dim = ['boolean', ['feature-state', 'dim'], false];
    // A trip is a band under the shortcuts it threads together, not another
    // line among them: the point of turning it on is to see which shortcuts a
    // walk strings together, so it has to be wide enough to show either side
    // of a five-pixel trail at every zoom, and pale enough to read through.
    // A fixed nine pixels is wider than a trail at z18 and invisible under one
    // at z13, which is where the first attempt at this went wrong.
    const band = layer.kind === 'trips';
    const paint = {
      'line-color': ['get', 'color'],
      'line-width': band
        ? ['interpolate', ['linear'], ['zoom'],
            12, ['case', sel, 10, 7],
            15, ['case', sel, 16, 12],
            18, ['case', sel, 26, 20]]
        : ['case', sel, 8, layer.kind === 'network' ? 3.5 : 5],
      // Unselected lines stay clearly readable: picking one trail should not
      // stop you browsing straight on to the next one from the map.
      'line-opacity': ['case', sel, band ? 0.75 : 1, dim, band ? 0.22 : 0.55,
        layer.kind === 'network' ? 0.75 : band ? 0.55 : 0.82]
    };
    // A dashed line reads as "not there yet" for the proposed network, and as
    // "not published yet" for a draft.
    if (layer.dash) paint['line-dasharray'] = [2, 1.6];

    map.addLayer({
      id: lineId(layer.id),
      type: 'line',
      source: src,
      layout: { 'line-cap': 'round', 'line-join': 'round' },
      paint
    });

    // A fat transparent line on top, so a fingertip does not have to land on
    // the stroke itself.
    map.addLayer({
      id: hitId(layer.id),
      type: 'line',
      source: src,
      paint: { 'line-color': '#000', 'line-opacity': 0, 'line-width': 22 }
    });
  }

  /** Build every source and layer. Re-run after each style change, because
   *  setStyle drops anything we added. */
  function addToMap() {
    if (typeof map === 'undefined' || !map) return;
    feature.clear();

    // A layer deleted since the last run leaves its GL layers and its source
    // behind. Sources go last: removing one still in use throws.
    const live = new Set(list.map((l) => l.id));
    const gone = new Set();
    map.getStyle().layers.forEach((gl) => {
      const match = /^(?:ln|pt|lb|hit)-(.+)$/.exec(gl.id);
      if (match && !live.has(match[1]) && map.getLayer(gl.id)) {
        map.removeLayer(gl.id);
        gone.add(match[1]);
      }
    });
    gone.forEach((id) => {
      if (map.getSource(srcId(id))) map.removeSource(srcId(id));
      wired.delete(id);
    });

    list.forEach((layer) => {
      // Nothing of a waypoints layer is drawn through WebGL - its pins are HTML
      // markers that app.js places from markerWaypoints - so it wants neither a
      // source nor a layer, and an empty line layer would only sit there
      // collecting click handlers for features that never exist.
      if (layer.kind === 'waypoints') return;

      const src = srcId(layer.id);
      if (map.getSource(src)) {
        map.getSource(src).setData(geojson(layer));
      } else {
        map.addSource(src, { type: 'geojson', data: geojson(layer) });
      }
      if (map.getLayer(hitId(layer.id))) return;

      // Areas first, so the dots and their labels land on top of them.
      if (hasShapes(layer)) addShapeLayers(layer);
      if (layer.kind === 'places') addPlaceLayers(layer);
      else addLineLayers(layer);

      // A layer-scoped listener outlives the layer it names, and setStyle wipes
      // every layer we added - so without this set, each basemap switch bound
      // another copy of the same three handlers.
      if (wired.has(layer.id)) return;
      wired.add(layer.id);
      map.on('click', hitId(layer.id), (e) => {
        if (e.features && e.features.length) select(e.features[0].properties.id, false);
      });
      map.on('mouseenter', hitId(layer.id), () => { map.getCanvas().style.cursor = 'pointer'; });
      map.on('mouseleave', hitId(layer.id), () => { map.getCanvas().style.cursor = ''; });
    });

    applyVisibility();
  }

  /* While the arrange tool is open it draws its own draggable marker for every
   * place, and the layer's dots would sit underneath every one of them. */
  let arranging = false;
  const setArranging = (value) => { arranging = value; applyVisibility(); };


  function applyVisibility() {
    // Ahead of the map guard: the legend is HTML, and it has to be right on a
    // first load that reaches here before the style has finished loading.
    renderLegend();
    if (typeof map === 'undefined' || !map) return;
    list.forEach((layer) => {
      const hide = (arranging && layer.pinnable)
        || (layer.kind === 'pending' && !Store.isEditor());
      const v = layer.on && !hide ? 'visible' : 'none';
      drawnIds(layer).forEach((id) => {
        if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', v);
      });
    });
  }

  /** Push a layer's items back into its map source, after drafts change. */
  function refresh(id) {
    const layer = byId(id);
    if (!layer) return;
    reindex();
    if (typeof map !== 'undefined' && map && map.getSource(srcId(id))) {
      map.getSource(srcId(id)).setData(geojson(layer));
    }
    onChange();
  }

  /** Highlight one item and dim the rest, across every layer. */
  function highlight(id) {
    if (typeof map === 'undefined' || !map) return;
    [feature, shapeFeature].forEach((where) => {
      where.forEach(({ src, fid }, ourId) => {
        if (!map.getSource(src)) return;
        map.setFeatureState({ source: src, id: fid },
          { sel: ourId === id, dim: id != null && ourId !== id });
      });
    });
  }

  /* ---------- categories ----------
   *
   * Eleven layers is more than a flat list can carry, and they are not eleven
   * of the same thing: the walking layers are what the app is for, the places
   * are things somebody else wrote up, and the planning and cadastre layers are
   * reference material you consult and put away. Three sections that fold, the
   * way a file tree folds.
   *
   * Only the walking one is open on arrival. Somebody who opens the sheet is
   * usually there to turn a shortcut layer on or off, and the other eight are
   * one tap away rather than eight rows of scrolling. */

  const CATEGORIES = [
    { id: 'trails', name: 'שבילים', note: 'מה שהולכים בו' },
    { id: 'places', name: 'מקומות', note: 'מה שיש בדרך' },
    { id: 'other', name: 'אחרים', note: 'תכנון וקדסטר' }
  ];

  const CAT_PREF = 'dk.cats.v1';

  /* Open on arrival: the walking layers only, exactly as the other two are
   * remembered per browser once somebody has folded or unfolded them. */
  const catOpen = (() => {
    const fresh = { trails: true, places: false, other: false };
    try {
      const kept = JSON.parse(localStorage.getItem(CAT_PREF) || 'null');
      return kept && typeof kept === 'object' ? { ...fresh, ...kept } : fresh;
    } catch (err) {
      return fresh;
    }
  })();

  function saveCats() {
    try {
      localStorage.setItem(CAT_PREF, JSON.stringify(catOpen));
    } catch (err) {
      /* private mode: the sections simply open at their defaults next time */
    }
  }

  /** Which section a layer belongs to. A layer that names none of them is a
   *  walking layer, because that is what a new trail layer always is. */
  const catOf = (layer) => (
    CATEGORIES.some((c) => c.id === layer.category) ? layer.category : 'trails');

  /* ---------- the legend ---------- */

  /* A layer is drawn in one colour only when its source has nothing to sort its
   * points by. The four places layers all do have something - the appendix's
   * seven narratives, pardespedia's eight kinds of place - and the layer sheet
   * showed a single swatch for each, so the colours on the map meant nothing to
   * anybody who had not read the builder. This is the key to them. */

  const LEGEND_PREF = 'dk.legend.open.v1';

  let legendOpen = (() => {
    try {
      return localStorage.getItem(LEGEND_PREF) === '1';
    } catch (err) {
      return false;
    }
  })();

  /** What one layer contributes: its groups when its source sorts its points
   *  into any, otherwise the single colour it is drawn in.
   *
   *  Only groups with a point actually on the map are listed. A group whose
   *  places are all still unplaced - most of מקום שמור - would otherwise be a
   *  colour in the key that appears nowhere on the map. */
  function legendRows(layer) {
    const placed = layer.waypoints.filter((p) => !p.unplaced);
    // Segments as well as places, because trips are sorted into groups too -
    // by how hard they are - and they are lines.
    const members = placed.concat(layer.segments);
    if ((layer.groups || []).length) {
      const rows = layer.groups
        .map((g) => ({
          name: g.name,
          color: g.color,
          line: layer.kind !== 'places',
          n: members.filter((m) => m.group === g.name).length
        }))
        .filter((r) => r.n);
      // A grouped layer whose members have all been left ungrouped still has
      // to appear, or the colour on the map answers to nothing.
      if (rows.length) return rows;
    }
    if (!layer.segments.length && !placed.length) return [];
    // A trail is a line on the map, a place is a dot, and a layer drawn as
    // areas is the edge of one. The key has to be the same shape as the thing
    // it stands for.
    return [{
      name: layer.name,
      color: layer.color,
      line: layer.kind !== 'places' || hasShapes(layer),
      dash: layer.dash,
      whole: true
    }];
  }

  function renderLegend() {
    const box = document.getElementById('legend');
    if (!box) return;

    const all = list.slice().reverse().filter(shown)
      .map((layer) => ({ layer, rows: legendRows(layer) }))
      .filter((s) => s.rows.length);

    // Lines first, then the layers that break into groups. Interleaved by draw
    // order they read as a jumble: a bare row, a heading and its seven colours,
    // then another bare row with nothing to say which layer it belonged to.
    const sections = [...all.filter((s) => s.rows[0].whole),
      ...all.filter((s) => !s.rows[0].whole)];

    box.hidden = !sections.length;
    if (!sections.length) return;

    const approx = sections.some((s) =>
      s.layer.waypoints.some((p) => p.vague && !p.unplaced));

    document.getElementById('legend-body').innerHTML = sections.map(({ layer, rows }) => {
      // A one-colour layer names itself in its only row, so a heading above it
      // would just say the same thing twice.
      const head = rows[0].whole ? ''
        : `<p class="lg-layer">${escapeHtml(layer.name)}</p>`;
      return head + `<ul class="lg-rows">${rows.map((r) => `<li>
        <span class="lg-dot${r.line ? ' line' : ''}${r.dash ? ' dash' : ''}"
              style="--c:${r.color}"></span>
        <span class="lg-nm">${escapeHtml(r.name)}</span>
        ${r.n ? `<span class="lg-n">${r.n}</span>` : ''}
      </li>`).join('')}</ul>`;
    }).join('') + (approx ? `<p class="lg-foot">
      <span class="lg-dot approx" style="--c:#667571"></span>
      עיגול חיוור בטבעת צבעונית: מיקום מקורב, לפי הרחוב בלבד, לפי מקום סמוך או
      לפי חלקה שכנה. הנקודה בסביבה הנכונה ולא על המקום עצמו.</p>` : '');

    document.getElementById('legend-body').hidden = !legendOpen;
    document.getElementById('legend-btn')
      .setAttribute('aria-expanded', legendOpen ? 'true' : 'false');
    box.classList.toggle('open', legendOpen);
  }

  function toggleLegend() {
    legendOpen = !legendOpen;
    try {
      localStorage.setItem(LEGEND_PREF, legendOpen ? '1' : '0');
    } catch (err) {
      /* private mode: it simply opens closed next time */
    }
    renderLegend();
  }

  /* ---------- the layer sheet ---------- */

  function summary(layer) {
    if (layer.kind === 'places') {
      const n = layer.waypoints.length;
      const missing = layer.waypoints.filter((p) => p.unplaced).length;
      return `${n} ${layer.unit || 'מקומות'}`
        + (missing ? ` · ${missing} עוד לא ממוקמים` : '');
    }
    // A trip is one walk, not a run of segments, so it is counted as itself.
    if (layer.kind === 'trips') {
      const metresTotal = layer.segments.reduce((sum, t) => sum + (t.length || 0), 0);
      const broken = layer.segments.filter((t) => (t.missing || []).length).length;
      if (!layer.segments.length) return 'אין עדיין טיולים. אפשר להוסיף אחד.';
      return (layer.segments.length === 1 ? 'טיול אחד'
        : `${layer.segments.length} ${layer.unit || 'טיולים'}`)
        + (metresTotal ? ` · ${(metresTotal / 1000).toFixed(1)} ק"מ סך הכל` : '')
        + (broken ? ` · ${broken} עם שביל חסר` : '');
    }

    if (layer.kind === 'waypoints') {
      const n = layer.waypoints.length;
      const shot = layer.waypoints.reduce((sum, w) => sum + (w.photos || []).length, 0);
      if (!n) return 'ריק';
      return `${n === 1 ? 'מקום אחד' : `${n} ${layer.unit || 'מקומות'}`}`
        + (shot ? ` · ${shot} תמונות` : '');
    }

    const n = layer.segments.length + layer.waypoints.length;
    if (!n) {
      if (layer.kind === 'drafts') return 'אין עדיין. הקלט או צייר שביל.';
      // "ריק" on the queue reads like something is broken. It is the ordinary
      // state: nobody has sent a trail in since the last one was dealt with.
      if (layer.kind === 'pending') return 'אין שביל שממתין כרגע.';
      return 'ריק';
    }
    const metres = layer.segments.reduce((sum, s) => sum + (s.length || 0), 0);
    const bits = [`${layer.segments.length} מקטעים`];
    if (metres) bits.push(metres >= 1000 ? (metres / 1000).toFixed(1) + ' ק"מ' : metres + ' מ׳');
    if (layer.waypoints.length) bits.push(`${layer.waypoints.length} נקודות`);
    return bits.join(' · ');
  }

  /** One layer's row, and whatever hangs off it. */
  function layerRow(layer, editable) {
    const rows = legendRows(layer);
    return `
      <div class="lay-row">
        <label class="lay${layer.on ? ' on' : ''}" data-id="${layer.id}">
          <input type="checkbox" ${layer.on ? 'checked' : ''}>
          <span class="lay-swatch${layer.dash ? ' dash' : ''}"
                style="--c:${layer.color}"></span>
          <span class="lay-txt">
            <span class="lay-nm">${escapeHtml(layer.name)}</span>
            <span class="lay-sub">${escapeHtml(summary(layer))}</span>
            <span class="lay-note">${escapeHtml(layer.note || '')}</span>
          </span>
        </label>
        ${editable && layer.own ? `<button class="lay-edit" data-edit="${layer.id}"
          aria-label="עריכת השכבה ${escapeHtml(layer.name)}">עריכה</button>` : ''}
      </div>
      ${layer.on && rows.length && !rows[0].whole ? `
        <ul class="lay-legend">${rows.map((r) => `<li>
          <span class="lg-dot" style="--c:${r.color}"></span>${escapeHtml(r.name)}
        </li>`).join('')}</ul>` : ''}
      ${editable && layer.pinnable ? `
        <button class="lay-add places" data-arrange="1">סידור מיקומי המקומות${
          layer.waypoints.filter((p) => p.unplaced).length
            ? ` · ${layer.waypoints.filter((p) => p.unplaced).length} עוד לא ממוקמים` : ''}
        </button>` : ''}`;
  }

  /** What a folded section says about itself.
   *
   *  The count and the swatches matter more than they look: without them,
   *  somebody switches a layer on, folds the section, and has no way to tell
   *  from the sheet why the map is covered in pins. */
  function catHead(cat, members) {
    const on = members.filter((l) => l.on);
    const open = !!catOpen[cat.id];
    return `
      <button class="cat-head${open ? ' open' : ''}" data-cat="${cat.id}"
              aria-expanded="${open}" aria-controls="cat-${cat.id}">
        <span class="cat-tw" aria-hidden="true"></span>
        <span class="cat-txt">
          <span class="cat-nm">${escapeHtml(cat.name)}</span>
          <span class="cat-sub">${on.length
            ? `${on.length} מתוך ${members.length} דלוקות`
            : `${members.length} שכבות · ${escapeHtml(cat.note)}`}</span>
        </span>
        <span class="cat-dots" aria-hidden="true">${on.slice(0, 6).map((l) =>
          `<span class="lg-dot${l.dash ? ' line dash' : ''}" style="--c:${l.color}"></span>`
        ).join('')}</span>
      </button>`;
  }

  function render() {
    const box = document.getElementById('layer-list');
    const editable = Store.isEditor();

    const shownLayers = list.slice().reverse()
      .filter((layer) => layer.kind !== 'pending' || editable);

    box.innerHTML = CATEGORIES.map((cat) => {
      const members = shownLayers.filter((l) => catOf(l) === cat.id);
      if (!members.length) return '';
      const open = !!catOpen[cat.id];
      return `<section class="lay-cat${open ? ' open' : ''}">
        ${catHead(cat, members)}
        <div class="cat-body" id="cat-${cat.id}"${open ? '' : ' hidden'}>
          ${members.map((l) => layerRow(l, editable)).join('')}
          ${editable && cat.id === 'trails'
            ? '<button class="lay-add" data-newlayer="1">+ שכבת שבילים חדשה</button>'
            : ''}
        </div>
      </section>`;
    }).join('');

    // The headers summarise what is on inside them, so they go stale the moment
    // a checkbox moves. Repainting only the headers leaves the row somebody is
    // still looking at, and the scroll position, exactly where they were.
    const paintHeads = () => {
      box.querySelectorAll('.lay-cat').forEach((section) => {
        const head = section.querySelector('.cat-head');
        const cat = CATEGORIES.find((c) => c.id === head.dataset.cat);
        head.outerHTML = catHead(cat, shownLayers.filter((l) => catOf(l) === cat.id));
      });
    };

    box.querySelectorAll('.lay input').forEach((tick) => {
      tick.addEventListener('change', () => {
        const layer = byId(tick.closest('.lay').dataset.id);
        layer.on = tick.checked;
        tick.closest('.lay').classList.toggle('on', layer.on);
        savePrefs();
        applyVisibility();
        onChange();
        paintHeads();
      });
    });

    const credits = list.filter((l) => l.on && l.credit).map((l) => l.credit);
    document.getElementById('layer-credit').textContent =
      credits.length ? 'מקורות: ' + credits.join(' · ') : '';
  }

  /* Bound once on the container, so repainting a header - which replaces the
   * element - never costs it its click handler. */
  let foldWired = false;
  function wireFolding() {
    if (foldWired) return;
    foldWired = true;
    document.getElementById('layer-list').addEventListener('click', (e) => {
      const head = e.target.closest('.cat-head');
      if (!head) return;
      catOpen[head.dataset.cat] = !catOpen[head.dataset.cat];
      saveCats();
      render();
    });
  }

  function openSheet() {
    wireFolding();
    render();
    document.getElementById('layer-sheet').hidden = false;
  }

  const closeSheet = () => { document.getElementById('layer-sheet').hidden = true; };

  return {
    list, init, add, byId, item, layerOf, reindex, resetTrails, resetPlaces,
    visible, visibleSegments, visibleWaypoints, markerWaypoints, trailLayers, stats,
    addToMap, applyVisibility, refresh, highlight, setArranging, setPending,
    openSheet, closeSheet, render,
    TRAILS_ID, PLACES_ID, PENDING_ID, ART_ID, SHIMUR_ID, MAKOM_ID, PLANS_ID,
    BLOCKS_ID, TRIPS_ID, TRIP_GAP_M, DIFFICULTY,
    resolveTrip, toTrip, pathLength, metres, isLoop,
    trailHitLayers, turnOn, tripsUsing,
    set onChange(fn) { onChange = fn; }
  };
})();
